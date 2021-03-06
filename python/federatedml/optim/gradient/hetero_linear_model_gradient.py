#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import functools

import numpy as np
import random
import scipy.sparse as sp

from federatedml.feature.sparse_vector import SparseVector
from federatedml.statistic import data_overview
from federatedml.util import LOGGER
from federatedml.util import consts
from federatedml.util import fate_operator
from federatedml.util.fixpoint_solver import FixedPointEncoder
from federatedml.linear_model.linear_model_weight import LinearModelWeights

from federatedml.secureprotol.fate_paillier import PaillierEncryptedNumber

from federatedml.secureprotol.fixedpoint import FixedPointNumber

from fate_arch.abc import CTableABC


def is_table(v):
    return isinstance(v, CTableABC)


class HeteroGradientBase(object):
    def __init__(self):
        self.use_async = False
        self.use_sample_weight = False
        self.fixed_point_encoder = None
        # self.Q = 293973345475167247070445277780365744413
        self.psi = 0
        self.transfer_variables = None

    def compute_gradient_procedure(self, *args):
        raise NotImplementedError("Should not call here")

    def set_total_batch_nums(self, total_batch_nums):
        """
        Use for sqn gradient.
        """
        pass

    def set_use_async(self):
        self.use_async = True

    def set_use_sample_weight(self):
        self.use_sample_weight = True

    def set_fixed_float_precision(self, floating_point_precision):
        if floating_point_precision is not None:
            self.fixed_point_encoder = FixedPointEncoder(2**floating_point_precision)

    # label_table = data_instances.mapValues(lambda x: x.label)
    # Y is weight
    def share_protocol1_B(self, Y, cipher, remote_role, suffix):
        roles = [consts.GUEST, consts.HOST]
        if is_table(Y):
            encrypted_Y = Y.mapValues(lambda v: cipher.encrypt(v))
        else:
            encrypted_Y = LinearModelWeights(cipher.encrypt_list(Y.coef_), False)
        self.transfer_variables.share_protocol.remote(obj=encrypted_Y, role=remote_role, idx=-1, suffix=suffix)
        # encrypted_Z_share = self.transfer_variables.share_protocol.get(idx=roles.index(remote_role), suffix=suffix)
        # Z_share = encrypted_Z_share.mapValues(cipher.decrypt)
        # Z_share = Z_share.mapValues(lambda v: FixedPointNumber.encode(v, n=self.psi))
        Z_share = self.share_protocol2_B(cipher, remote_role, suffix)
        return Z_share

    def share_protocol1_A(self, X, remote_role, suffix, compute_gradient = False):
        roles = [consts.GUEST, consts.HOST]
        encrypted_Y = self.transfer_variables.share_protocol.get(idx=roles.index(remote_role), suffix=suffix)
        encrypted_Z = self.compute_Z(X, encrypted_Y, compute_gradient)
        # Z_share = encrypted_Z.mapValues(lambda v: FixedPointNumber.rand_number_generator(self.psi))
        # Z_fore = encrypted_Z.join(Z_share, lambda d, g: d - g)
        # self.transfer_variables.share_protocol.remote(obj=Z_fore, role=remote_role, idx=-1, suffix=suffix)
        Z_share = self.share_protocol2_A(encrypted_Z, remote_role, suffix)
        return Z_share

    def share_protocol2_B(self, cipher, remote_role, suffix):
        roles = [consts.GUEST, consts.HOST]
        encrypted_Z_share = self.transfer_variables.share_protocol.get(idx=roles.index(remote_role), suffix=suffix)
        if is_table(encrypted_Z_share):
            Z_share = encrypted_Z_share.mapValues(cipher.decrypt)
            Z_share = Z_share.mapValues(lambda v: FixedPointNumber.encode(v, n=self.psi))
        else:
            Z_share = cipher.decrypt_list(encrypted_Z_share)
            Z_share = np.array([FixedPointNumber.encode(v, n=self.psi) for v in Z_share])
        return Z_share

    def share_protocol2_A(self, encrypted_Z, remote_role, suffix):
        roles = [consts.GUEST, consts.HOST]
        if is_table(encrypted_Z):
            Z_share = encrypted_Z.mapValues(lambda v: FixedPointNumber.rand_number_generator(self.psi))
            Z_fore = encrypted_Z.join(Z_share, lambda d, g: d - g)
        else:
            Z_share = np.array([FixedPointNumber.rand_number_generator(self.psi) for v in encrypted_Z])
            Z_fore = encrypted_Z - Z_share
        self.transfer_variables.share_protocol.remote(obj=Z_fore, role=remote_role, idx=-1, suffix=suffix)
        return Z_share

    def compute_Z(self, X, Y, compute_gradient=False):
        """
        Z = X[Y]
        """
        if compute_gradient:
            Z = self.compute_gradient(X, Y, False)
        else:
            Z = X.mapValues(lambda v: fate_operator.vec_dot(v.features, Y.coef_))
        return Z


    @staticmethod
    def __compute_partition_gradient(data, fit_intercept=True, is_sparse=False):
        """
        Compute hetero regression gradient for:
        gradient = ???d*x, where d is fore_gradient which differ from different algorithm
        Parameters
        ----------
        data: DTable, include fore_gradient and features
        fit_intercept: bool, if model has interception or not. Default True

        Returns
        ----------
        numpy.ndarray
            hetero regression model gradient
        """
        feature = []
        fore_gradient = []

        if is_sparse:
            row_indice = []
            col_indice = []
            data_value = []

            row = 0
            feature_shape = None
            for key, (sparse_features, d) in data:
                fore_gradient.append(d)
                assert isinstance(sparse_features, SparseVector)
                if feature_shape is None:
                    feature_shape = sparse_features.get_shape()
                for idx, v in sparse_features.get_all_data():
                    col_indice.append(idx)
                    row_indice.append(row)
                    data_value.append(v)
                row += 1
            if feature_shape is None or feature_shape == 0:
                return 0
            sparse_matrix = sp.csr_matrix((data_value, (row_indice, col_indice)), shape=(row, feature_shape))
            fore_gradient = np.array(fore_gradient)

            # gradient = sparse_matrix.transpose().dot(fore_gradient).tolist()
            gradient = fate_operator.dot(sparse_matrix.transpose(), fore_gradient).tolist()
            if fit_intercept:
                bias_grad = np.sum(fore_gradient)
                gradient.append(bias_grad)
                # LOGGER.debug("In first method, gradient: {}, bias_grad: {}".format(gradient, bias_grad))
            return np.array(gradient)

        else:
            for key, value in data:
                feature.append(value[0])
                fore_gradient.append(value[1])
            feature = np.array(feature)
            fore_gradient = np.array(fore_gradient)
            if feature.shape[0] <= 0:
                return 0

            gradient = fate_operator.dot(feature.transpose(), fore_gradient)
            gradient = gradient.tolist()
            if fit_intercept:
                bias_grad = np.sum(fore_gradient)
                gradient.append(bias_grad)
            return np.array(gradient)

    @staticmethod
    def __apply_cal_gradient(data, fixed_point_encoder, is_sparse):
        all_g = None
        for key, (feature, d) in data:
            if is_sparse:
                x = np.zeros(feature.get_shape())
                for idx, v in feature.get_all_data():
                    x[idx] = v
                feature = x
            if fixed_point_encoder:
                # g = (feature * 2 ** floating_point_precision).astype("int") * d
                g = fixed_point_encoder.encode(feature) * d
                
                # if isinstance(d, PaillierEncryptedNumber):
                #     LOGGER.info("---->ForeGradient: d {}".format(d.exponent))
                #     LOGGER.info("---->Gradient: g {}".format(g[0].exponent))
            else:
                g = feature * d
            if all_g is None:
                all_g = g
            else:
                all_g += g
        if all_g is None:
            return all_g
        elif fixed_point_encoder:
            # if isinstance(all_g[0], PaillierEncryptedNumber):
            #     LOGGER.info("---->Gradient: before div decode all_g {}".format(all_g[0].exponent))
            all_g = fixed_point_encoder.decode(all_g)
            # if isinstance(all_g[0], PaillierEncryptedNumber):
            #     LOGGER.info("---->Gradient: after div decode all_g {}".format(all_g[0].exponent))
        return all_g

    def compute_gradient(self, data_instances, fore_gradient, fit_intercept):
        """
        Compute hetero-regression gradient
        Parameters
        ----------
        data_instances: DTable, input data
        fore_gradient: DTable, fore_gradient
        fit_intercept: bool, if model has intercept or not

        Returns
        ----------
        DTable
            the hetero regression model's gradient
        """

        # print("------>>hahah", type(fore_gradient))

        # for x in fore_gradient:
        #     # LOGGER.info("---->ForeGradient: Type {} value {}".format(type(x), x))
        #     if isinstance(x, PaillierEncryptedNumber):
        #         LOGGER.info("---->ForeGradient: exponent {}".format(x.exponent))

        feature_num = data_overview.get_features_shape(data_instances)
        data_count = data_instances.count()
        is_sparse = data_overview.is_sparse_data(data_instances)

        if data_count * feature_num > 100:
            LOGGER.debug("Use apply partitions")
            feat_join_grad = data_instances.join(fore_gradient,
                                                 lambda d, g: (d.features, g))
            f = functools.partial(self.__apply_cal_gradient,
                                  fixed_point_encoder=self.fixed_point_encoder,
                                  is_sparse=is_sparse)
            gradient_sum = feat_join_grad.applyPartitions(f)
            # LOGGER.info("hahahha1--> {}".format(gradient_sum.take(1)[0]))
            # if isinstance(gradient_sum.take(1)[0][1][0], PaillierEncryptedNumber):
            #     LOGGER.info("---->gradient_sum.take(1)[0][1][0]: exponent {}".format(gradient_sum.take(1)[0][1][0].exponent))
            
            gradient_sum = gradient_sum.reduce(lambda x, y: x + y)
            # LOGGER.info("hahahha2--> {}".format(type(gradient_sum)))
            if fit_intercept:
                # bias_grad = np.sum(fore_gradient)
                bias_grad = fore_gradient.reduce(lambda x, y: x + y)
                gradient_sum = np.append(gradient_sum, bias_grad)

            # if isinstance(gradient_sum[0], PaillierEncryptedNumber):
            #     LOGGER.info("---->GradientSum[0]: exponent {}".format(gradient_sum[0].exponent))
            gradient = gradient_sum / data_count

        else:
            LOGGER.debug(f"Original_method")
            feat_join_grad = data_instances.join(fore_gradient,
                                                 lambda d, g: (d.features, g))
            f = functools.partial(self.__compute_partition_gradient,
                                  fit_intercept=fit_intercept,
                                  is_sparse=is_sparse)
            gradient_partition = feat_join_grad.applyPartitions(f)
            # if isinstance(gradient_partition[0][0], PaillierEncryptedNumber):
            #     LOGGER.info("---->GradientPartition[0]: exponent {}".format(gradient_partition[0].exponent))
            gradient_partition = gradient_partition.reduce(lambda x, y: x + y)

            gradient = gradient_partition / data_count

        # for x in gradient:
        #     LOGGER.info("---->Gradient: Type {} value {}".format(type(x), x))
        #     if isinstance(x, PaillierEncryptedNumber):
        #         LOGGER.info("---->Gradient: exponent {}".format(x.exponent))

        return gradient


class Guest(HeteroGradientBase):
    def __init__(self):
        super().__init__()
        self.half_d = None
        self.host_forwards = None
        self.forwards = None
        self.aggregated_forwards = None

    def _register_gradient_sync(self, host_forward_transfer, fore_gradient_transfer,
                                guest_gradient_transfer, guest_optim_gradient_transfer):
        self.host_forward_transfer = host_forward_transfer
        self.fore_gradient_transfer = fore_gradient_transfer
        self.unilateral_gradient_transfer = guest_gradient_transfer
        self.unilateral_optim_gradient_transfer = guest_optim_gradient_transfer

    def compute_and_aggregate_forwards(self, data_instances, model_weights,
                                       encrypted_calculator, batch_index, current_suffix, offset=None):
        raise NotImplementedError("Function should not be called here")

    def compute_half_d(self, data_instances, w, cipher, batch_index, current_suffix):
        raise NotImplementedError("Function should not be called here")

    def _asynchronous_compute_gradient(self, data_instances, model_weights, cipher, current_suffix):
        LOGGER.debug("Called asynchronous gradient")
        encrypted_half_d = cipher.encrypt(self.half_d)
        self.remote_fore_gradient(encrypted_half_d, suffix=current_suffix)

        half_g = self.compute_gradient(data_instances, self.half_d, False)
        self.host_forwards = self.get_host_forward(suffix=current_suffix)
        host_forward = self.host_forwards[0]
        host_half_g = self.compute_gradient(data_instances, host_forward, False)
        unilateral_gradient = half_g + host_half_g
        if model_weights.fit_intercept:
            n = data_instances.count()
            intercept = (host_forward.reduce(lambda x, y: x + y) + self.half_d.reduce(lambda x, y: x + y)) / n
            unilateral_gradient = np.append(unilateral_gradient, intercept)
        return unilateral_gradient

    def _centralized_compute_gradient(self, data_instances, model_weights, cipher, current_suffix):
        self.host_forwards = self.get_host_forward(suffix=current_suffix)
        fore_gradient = self.half_d
        for host_forward in self.host_forwards:
            fore_gradient = fore_gradient.join(host_forward, lambda x, y: x + y)
        self.remote_fore_gradient(fore_gradient, suffix=current_suffix)
        unilateral_gradient = self.compute_gradient(data_instances, fore_gradient, model_weights.fit_intercept)
        return unilateral_gradient

    def compute_gradient_procedure(self, data_instances, guest_encrypted_calculator, host_encrypted_calculator, 
                                    guest_model_weights_guest_share, host_model_weights_guest_share, self_optimizer, remote_optimizer,
                                   n_iter_, batch_index, offset=None):
        """
          Linear model gradient procedure
          Step 1: get host forwards which differ from different algorithm
                  For Logistic Regression and Linear Regression: forwards = wx
                  For Poisson Regression, forwards = exp(wx)

          Step 2: Compute self forwards and aggregate host forwards and get d = fore_gradient

          Step 3: Compute unilateral gradient = ???d*x,

          Step 4: Send unilateral gradients to arbiter and received the optimized and decrypted gradient.
        """

        self.psi = guest_encrypted_calculator.public_key.n

        current_suffix = (n_iter_, batch_index, 0)

        host_z_guest_share = self.share_protocol1_B(host_model_weights_guest_share, guest_encrypted_calculator, consts.HOST, current_suffix)

        guest_z_guest_share = self.compute_Z(data_instances, guest_model_weights_guest_share)

        current_suffix = (n_iter_, batch_index, 1)

        guest_z_guest_share_new = self.share_protocol1_A(data_instances, consts.HOST, current_suffix)
        guest_z_guest_share = guest_z_guest_share.join(guest_z_guest_share_new, lambda d, g: d + g)

        z_guest_share = guest_z_guest_share.join(host_z_guest_share, lambda d, g: d + g)
        z_guest_share_square = z_guest_share.mapValues(lambda v: v * v)
        z_guest_share_cube = z_guest_share.mapValues(lambda v: v * v * v)

        roles = [consts.GUEST, consts.HOST]
        current_suffix = (n_iter_, batch_index, 2, 1)
        enc_z_host_share  = self.transfer_variables.share_protocol.get(idx=roles.index(consts.HOST), suffix=current_suffix)
        current_suffix = (n_iter_, batch_index, 2, 2)
        enc_z_host_share_square = self.transfer_variables.share_protocol.get(idx=roles.index(consts.HOST), suffix=current_suffix)
        current_suffix = (n_iter_, batch_index, 2, 3)
        enc_z_host_share_cube  = self.transfer_variables.share_protocol.get(idx=roles.index(consts.HOST), suffix=current_suffix)

        # LOGGER.debug(f"z_guest_share type {type(z_guest_share._table)}")
        # LOGGER.debug(f"enc_z_host_share_square type {type(enc_z_host_share_square._table)}")

        enc_z_cube_1 = enc_z_host_share_square.join(z_guest_share, lambda d, g: d * g * 3)
        enc_z_cube_2 = enc_z_host_share.join(z_guest_share_square, lambda d, g: d * g * 3)
        enc_z_cube = enc_z_host_share_cube.join(z_guest_share_cube, lambda d, g: d + g)
        enc_z_cube = enc_z_cube.join(enc_z_cube_1, lambda d, g: d + g)
        enc_z_cube = enc_z_cube.join(enc_z_cube_2, lambda d, g: d + g)
        enc_z = enc_z_host_share.join(z_guest_share, lambda d, g: d + g)

        enc_y_hat_minimax = enc_z_cube.join(enc_z, lambda d, g: d * (-0.004) + g * (0.197) + 0.5)
        # sigmoid_z = complete_z * 0.25 + 0.5
        enc_y_hat_taylor = enc_z_cube.join(enc_z, lambda d, g: g * (0.25) + 0.5)

        enc_y_hat = enc_z_cube.join(enc_z, lambda d, g: d * (-0.004) + g * (0.197) + 0.5)

        # Test reveal enc_y_hat
        current_suffix = (n_iter_, batch_index, "test enc_y_hat_minimax")
        enc_y_hat_minimax = enc_y_hat_minimax.join(enc_z, lambda d, g: [g, d])
        self.transfer_variables.share_protocol.remote(obj=enc_y_hat_minimax, role=consts.HOST, idx=-1, suffix=current_suffix)

        current_suffix = (n_iter_, batch_index, "test enc_y_hat_Taylor")
        enc_y_hat_taylor = enc_y_hat_taylor.join(enc_z, lambda d, g: [g, d])
        self.transfer_variables.share_protocol.remote(obj=enc_y_hat_taylor, role=consts.HOST, idx=-1, suffix=current_suffix)

        tmp_labels = data_instances.mapValues(lambda v: v.label)
        LOGGER.debug("tmp_labels: ")
        tmp_labels = tmp_labels.take(10)
        for x in tmp_labels:
            LOGGER.debug(f"{x[1]}")        

        enc_e = enc_y_hat.join(data_instances, lambda d, g: d - g.label)

        current_suffix = (n_iter_, batch_index, 3)
        y_hat_guest_share = self.share_protocol2_A(enc_y_hat, consts.HOST, current_suffix)

        e_guest_share = y_hat_guest_share.join(data_instances, lambda d, g: d - g.label)

        enc_guest_gradient = self.compute_gradient(data_instances, enc_e, False)

        current_suffix = (n_iter_, batch_index, 4)
        guest_gradient_guest_share = self.share_protocol2_A(enc_guest_gradient, consts.HOST, current_suffix)

        current_suffix = (n_iter_, batch_index, 5)
        host_gradient_guest_share = self.share_protocol1_B(e_guest_share, guest_encrypted_calculator, consts.HOST, current_suffix)

        # Apply gradient learning rate
        host_delta_grad_guest_share = remote_optimizer.apply_gradients(host_gradient_guest_share)
        guest_delta_grad_guest_share = self_optimizer.apply_gradients(guest_gradient_guest_share)

        return (guest_delta_grad_guest_share, host_delta_grad_guest_share)
        
        # # self.host_forwards = self.get_host_forward(suffix=current_suffix)

        # # Compute Guest's partial d
        # self.compute_half_d(data_instances, model_weights, encrypted_calculator,
        #                     batch_index, current_suffix)
        # if self.use_async:
        #     unilateral_gradient = self._asynchronous_compute_gradient(data_instances, model_weights,
        #                                                               cipher=encrypted_calculator[batch_index],
        #                                                               current_suffix=current_suffix)
        # else:
        #     unilateral_gradient = self._centralized_compute_gradient(data_instances, model_weights,
        #                                                              cipher=encrypted_calculator[batch_index],
        #                                                              current_suffix=current_suffix)

        # if optimizer is not None:
        #     unilateral_gradient = optimizer.add_regular_to_grad(unilateral_gradient, model_weights)

        # optimized_gradient = self.update_gradient(unilateral_gradient, suffix=current_suffix)
        # # LOGGER.debug(f"Before return, optimized_gradient: {optimized_gradient}")
        # return optimized_gradient

    def get_host_forward(self, suffix=tuple()):
        host_forward = self.host_forward_transfer.get(idx=-1, suffix=suffix)
        return host_forward

    def remote_fore_gradient(self, fore_gradient, suffix=tuple()):
        self.fore_gradient_transfer.remote(obj=fore_gradient, role=consts.HOST, idx=-1, suffix=suffix)

    def update_gradient(self, unilateral_gradient, suffix=tuple()):
        self.unilateral_gradient_transfer.remote(unilateral_gradient, role=consts.ARBITER, idx=0, suffix=suffix)
        optimized_gradient = self.unilateral_optim_gradient_transfer.get(idx=0, suffix=suffix)
        return optimized_gradient


class Host(HeteroGradientBase):
    def __init__(self):
        super().__init__()
        self.forwards = None
        self.fore_gradient = None

    def _register_gradient_sync(self, host_forward_transfer, fore_gradient_transfer,
                                host_gradient_transfer, host_optim_gradient_transfer):
        self.host_forward_transfer = host_forward_transfer
        self.fore_gradient_transfer = fore_gradient_transfer
        self.unilateral_gradient_transfer = host_gradient_transfer
        self.unilateral_optim_gradient_transfer = host_optim_gradient_transfer

    def compute_forwards(self, data_instances, model_weights):
        raise NotImplementedError("Function should not be called here")

    def compute_unilateral_gradient(self, data_instances, fore_gradient, model_weights, optimizer):
        raise NotImplementedError("Function should not be called here")

    def _asynchronous_compute_gradient(self, data_instances, cipher, current_suffix):
        encrypted_forward = cipher.encrypt(self.forwards)
        self.remote_host_forward(encrypted_forward, suffix=current_suffix)

        half_g = self.compute_gradient(data_instances, self.forwards, False)
        guest_half_d = self.get_fore_gradient(suffix=current_suffix)
        guest_half_g = self.compute_gradient(data_instances, guest_half_d, False)
        unilateral_gradient = half_g + guest_half_g
        return unilateral_gradient

    def _centralized_compute_gradient(self, data_instances, cipher, current_suffix):
        encrypted_forward = cipher.encrypt(self.forwards)
        self.remote_host_forward(encrypted_forward, suffix=current_suffix)

        fore_gradient = self.fore_gradient_transfer.get(idx=0, suffix=current_suffix)

        # Host case, never fit-intercept
        unilateral_gradient = self.compute_gradient(data_instances, fore_gradient, False)
        return unilateral_gradient

    def compute_gradient_procedure(self, data_instances, host_encrypted_calculator, guest_encrypted_calculator, 
                                    host_model_weights_host_share, guest_model_weights_host_share,
                                   self_optimizer, remote_optimizer,
                                   n_iter_, batch_index):
        """
        Linear model gradient procedure
        Step 1: get host forwards which differ from different algorithm
                For Logistic Regression: forwards = wx


        """
        guest_encrypted_calculator = guest_encrypted_calculator[batch_index]
        self.psi = guest_encrypted_calculator.encrypter.public_key.n

        host_z_host_share = self.compute_Z(data_instances, host_model_weights_host_share)

        current_suffix = (n_iter_, batch_index, 0)

        host_z_host_share_new = self.share_protocol1_A(data_instances, consts.GUEST, current_suffix)
        host_z_host_share = host_z_host_share.join(host_z_host_share_new, lambda d, g: d + g)

        current_suffix = (n_iter_, batch_index, 1)

        guest_z_host_share = self.share_protocol1_B(guest_model_weights_host_share, host_encrypted_calculator, consts.GUEST, current_suffix)

        z_host_share = host_z_host_share.join(guest_z_host_share, lambda d, g: d + g)

        enc_z_host_share = z_host_share.mapValues(lambda v: host_encrypted_calculator.encrypt(v))
        enc_z_host_share_square = z_host_share.mapValues(lambda v: host_encrypted_calculator.encrypt(v * v))
        enc_z_host_share_cube = z_host_share.mapValues(lambda v: host_encrypted_calculator.encrypt(v * v * v))

        current_suffix = (n_iter_, batch_index, 2, 1)
        self.transfer_variables.share_protocol.remote(obj=enc_z_host_share, role=consts.GUEST, idx=-1, suffix=current_suffix)
        current_suffix = (n_iter_, batch_index, 2, 2)
        self.transfer_variables.share_protocol.remote(obj=enc_z_host_share_square, role=consts.GUEST, idx=-1, suffix=current_suffix)
        current_suffix = (n_iter_, batch_index, 2, 3)
        self.transfer_variables.share_protocol.remote(obj=enc_z_host_share_cube, role=consts.GUEST, idx=-1, suffix=current_suffix)

        # Test reveal enc_y_hat
        current_suffix = (n_iter_, batch_index, "test enc_y_hat_minimax")
        tmp_enc_y_hat_minimax = self.transfer_variables.share_protocol.get(idx=0, suffix=current_suffix)
        tmp_y_hat_minimax = tmp_enc_y_hat_minimax.mapValues(host_encrypted_calculator.decrypt_list)

        current_suffix = (n_iter_, batch_index, "test enc_y_hat_Taylor")
        tmp_enc_y_hat_taylor = self.transfer_variables.share_protocol.get(idx=0, suffix=current_suffix)
        tmp_y_hat_taylor = tmp_enc_y_hat_taylor.mapValues(host_encrypted_calculator.decrypt_list)

        LOGGER.debug("tmp_y_hat_minimax: ")
        tmp_y_hat_minimax = tmp_y_hat_minimax.take(10)
        for x in tmp_y_hat_minimax:
            LOGGER.debug(f"{x[1]}, {(-0.004 )* x[1][0] ** 3 + 0.197 * x[1][0] + 0.5}")

        LOGGER.debug("tmp_y_hat_Taylor: ")
        tmp_y_hat_taylor = tmp_y_hat_taylor.take(10)
        for x in tmp_y_hat_taylor:
            LOGGER.debug(f"{x[1]}, {0.25 * x[1][0] + 0.5}")
        

        current_suffix = (n_iter_, batch_index, 3)
        y_hat_host_share = self.share_protocol2_B(host_encrypted_calculator, consts.GUEST, current_suffix)
        e_host_share = y_hat_host_share.mapValues(lambda v:v)

        current_suffix = (n_iter_, batch_index, 4)
        guest_gradient_host_share = self.share_protocol2_B(host_encrypted_calculator, consts.GUEST, current_suffix)

        host_gradient_host_share = self.compute_gradient(data_instances, e_host_share, False)

        current_suffix = (n_iter_, batch_index, 5)

        host_gradient_host_share_new = self.share_protocol1_A(data_instances, consts.GUEST, current_suffix, compute_gradient=True)
        # host_gradient_host_share = host_gradient_host_share.join(host_gradient_host_share_new, lambda d, g: d + g)
        host_gradient_host_share = host_gradient_host_share + host_gradient_host_share_new

        # Apply gradient learning rate
        host_delta_grad_host_share = self_optimizer.apply_gradients(host_gradient_host_share)
        guest_delta_grad_host_share = remote_optimizer.apply_gradients(guest_gradient_host_share)

        return(host_delta_grad_host_share, guest_delta_grad_host_share)

        # self.forwards = self.compute_forwards(data_instances, model_weights)

        # if self.use_async:
        #     unilateral_gradient = self._asynchronous_compute_gradient(data_instances,
        #                                                               encrypted_calculator[batch_index],
        #                                                               current_suffix)
        # else:
        #     unilateral_gradient = self._centralized_compute_gradient(data_instances,
        #                                                              encrypted_calculator[batch_index],
        #                                                              current_suffix)

        # if optimizer is not None:
        #     unilateral_gradient = optimizer.add_regular_to_grad(unilateral_gradient, model_weights)

        # optimized_gradient = self.update_gradient(unilateral_gradient, suffix=current_suffix)
        # LOGGER.debug(f"Before return compute_gradient_procedure")
        # return optimized_gradient

    def compute_sqn_forwards(self, data_instances, delta_s, cipher_operator):
        """
        To compute Hessian matrix, y, s are needed.
        g = (1/N)*???(0.25 * wx - 0.5 * y) * x
        y = ???2^F(w_t)s_t = g' * s = (1/N)*???(0.25 * x * s) * x
        define forward_hess = ???(0.25 * x * s)
        """
        sqn_forwards = data_instances.mapValues(
            lambda v: cipher_operator.encrypt(fate_operator.vec_dot(v.features, delta_s.coef_) + delta_s.intercept_))
        # forward_sum = sqn_forwards.reduce(reduce_add)
        return sqn_forwards

    def compute_forward_hess(self, data_instances, delta_s, forward_hess):
        """
        To compute Hessian matrix, y, s are needed.
        g = (1/N)*???(0.25 * wx - 0.5 * y) * x
        y = ???2^F(w_t)s_t = g' * s = (1/N)*???(0.25 * x * s) * x
        define forward_hess = (0.25 * x * s)
        """
        hess_vector = self.compute_gradient(data_instances,
                                       forward_hess,
                                       delta_s.fit_intercept)
        return np.array(hess_vector)

    def remote_host_forward(self, host_forward, suffix=tuple()):
        self.host_forward_transfer.remote(obj=host_forward, role=consts.GUEST, idx=0, suffix=suffix)

    def get_fore_gradient(self, suffix=tuple()):
        host_forward = self.fore_gradient_transfer.get(idx=0, suffix=suffix)
        return host_forward

    def update_gradient(self, unilateral_gradient, suffix=tuple()):
        self.unilateral_gradient_transfer.remote(unilateral_gradient, role=consts.ARBITER, idx=0, suffix=suffix)
        optimized_gradient = self.unilateral_optim_gradient_transfer.get(idx=0, suffix=suffix)
        return optimized_gradient


class Arbiter(HeteroGradientBase):
    def __init__(self):
        super().__init__()
        self.has_multiple_hosts = False

    def _register_gradient_sync(self, guest_gradient_transfer, host_gradient_transfer,
                                guest_optim_gradient_transfer, host_optim_gradient_transfer):
        self.guest_gradient_transfer = guest_gradient_transfer
        self.host_gradient_transfer = host_gradient_transfer
        self.guest_optim_gradient_transfer = guest_optim_gradient_transfer
        self.host_optim_gradient_transfer = host_optim_gradient_transfer

    def compute_gradient_procedure(self, cipher_operator, optimizer, n_iter_, batch_index):
        """
        Compute gradients.
        Received local_gradients from guest and hosts. Merge and optimize, then separate and remote back.
        Parameters
        ----------
        cipher_operator: Use for encryption

        optimizer: optimizer that get delta gradient of this iter

        n_iter_: int, current iter nums

        batch_index: int, use to obtain current encrypted_calculator

        """
        current_suffix = (n_iter_, batch_index)

        host_gradients, guest_gradient = self.get_local_gradient(current_suffix)

        if len(host_gradients) > 1:
            self.has_multiple_hosts = True

        host_gradients = [np.array(h) for h in host_gradients]
        guest_gradient = np.array(guest_gradient)

        size_list = [h_g.shape[0] for h_g in host_gradients]
        size_list.append(guest_gradient.shape[0])

        gradient = np.hstack((h for h in host_gradients))
        gradient = np.hstack((gradient, guest_gradient))

        grad = np.array(cipher_operator.decrypt_list(gradient))

        # LOGGER.debug("In arbiter compute_gradient_procedure, before apply grad: {}, size_list: {}".format(
        #     grad, size_list
        # ))

        delta_grad = optimizer.apply_gradients(grad)

        # LOGGER.debug("In arbiter compute_gradient_procedure, delta_grad: {}".format(
        #     delta_grad
        # ))
        separate_optim_gradient = self.separate(delta_grad, size_list)
        # LOGGER.debug("In arbiter compute_gradient_procedure, separated gradient: {}".format(
        #     separate_optim_gradient
        # ))
        host_optim_gradients = separate_optim_gradient[: -1]
        guest_optim_gradient = separate_optim_gradient[-1]

        self.remote_local_gradient(host_optim_gradients, guest_optim_gradient, current_suffix)
        return delta_grad

    @staticmethod
    def separate(value, size_list):
        """
        Separate value in order to several set according size_list
        Parameters
        ----------
        value: list or ndarray, input data
        size_list: list, each set size

        Returns
        ----------
        list
            set after separate
        """
        separate_res = []
        cur = 0
        for size in size_list:
            separate_res.append(value[cur:cur + size])
            cur += size
        return separate_res

    def get_local_gradient(self, suffix=tuple()):
        host_gradients = self.host_gradient_transfer.get(idx=-1, suffix=suffix)
        LOGGER.info("Get host_gradient from Host")

        guest_gradient = self.guest_gradient_transfer.get(idx=0, suffix=suffix)
        LOGGER.info("Get guest_gradient from Guest")
        return host_gradients, guest_gradient

    def remote_local_gradient(self, host_optim_gradients, guest_optim_gradient, suffix=tuple()):
        for idx, host_optim_gradient in enumerate(host_optim_gradients):
            self.host_optim_gradient_transfer.remote(host_optim_gradient,
                                                     role=consts.HOST,
                                                     idx=idx,
                                                     suffix=suffix)

        self.guest_optim_gradient_transfer.remote(guest_optim_gradient,
                                                  role=consts.GUEST,
                                                  idx=0,
                                                  suffix=suffix)
