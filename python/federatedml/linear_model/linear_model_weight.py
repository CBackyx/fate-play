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

import random
import numpy as np
import math

from federatedml.framework.weights import ListWeights, TransferableWeights
from federatedml.secureprotol.fate_paillier import PaillierEncryptedNumber

from federatedml.secureprotol.fixedpoint import FixedPointNumber


class LinearModelWeights(ListWeights):
    def __init__(self, l, fit_intercept):
        # self.phi = 2**128
        # self.max_int = self.phi // 3 - 1
        self.precision = 2 ** 8
        self.Q = 293973345475167247070445277780365744413

        l = np.array(l)
        if (not isinstance(l[0], PaillierEncryptedNumber)) and (not isinstance(l[0], FixedPointNumber)):
            if np.max(np.abs(l)) > 1e8:
                raise RuntimeError("The model weights are overflow, please check if the "
                                   "input data has been normalized")
        super().__init__(l)
        self.fit_intercept = fit_intercept

    def for_remote(self):
        return TransferableWeights(self._weights, self.__class__, self.fit_intercept)

    @property
    def coef_(self):
        if self.fit_intercept:
            return np.array(self._weights[:-1])
        return np.array(self._weights)

    @property
    def intercept_(self):
        if self.fit_intercept:
            return self._weights[-1]
        return 0.0

    def binary_op(self, other: 'LinearModelWeights', func, inplace):
        if inplace:
            for k, v in enumerate(self._weights):
                self._weights[k] = func(self._weights[k], other._weights[k])
            return self
        else:
            _w = []
            for k, v in enumerate(self._weights):
                _w.append(func(self._weights[k], other._weights[k]))
            return LinearModelWeights(_w, self.fit_intercept)

    # def share_encode(self):
    #     if np.max(np.abs(self._weights)) > 1e8:
    #         raise RuntimeError("The model weights are overflow before encoding")        
    #     def encode(x):
    #         if x >= 0:
    #             x = math.floor(x * 1000)
    #         else:
    #             x = math.ceil(x * 1000) + self.phi
    #     self._weights = np.vectorize(encode)(self._weights)

    # def share_decode(self):
    #     if np.max(np.abs(self._weights)) >= self.phi:
    #         raise RuntimeError("The encoded model weights are unmoded") 
    #     def decode(x):
    #         if x <= (self.phi / 2):
    #             x = x / 1000
    #         else:
    #             x = (x - self.phi) / 1000
    #     self._weights = np.vectorize(decode)(self._weights)

    def share_encode(self):
        if np.max(np.abs(self._weights)) > 1e8:
            raise RuntimeError("The model weights are overflow before encoding")        
        def encode(x):
            # return FixedPointNumber.encode(x, self.phi, self.phi // 3 - 1, precision=self.precision)
            return FixedPointNumber.encode(x, precision=self.precision)
        self._weights = np.vectorize(encode)(self._weights)

    def share_decode(self):
        if np.max(np.abs(self._weights)) >= self.Q:
            raise RuntimeError("The encoded model weights are unmoded") 
        def decode(x):
            return x.decode()
        self._weights = np.vectorize(decode)(self._weights)

    def share(self):
        share_weights = np.array([FixedPointNumber(random.randint(0, self.Q - 1), x.exponent) for x in self._weights])
        # for i in range(self._weights.shape[0]):
        #     self._weights[i].encoding = (self._weights[i].encoding - share_weights[i].encoding) % self.Q
        # self._weights = np.array(self._weights - share_weights)
        self._weights = self._weights - share_weights
        return LinearModelWeights(share_weights, self.fit_intercept)

    def add_share(self, share_model_weights):
        self._weights = (self._weights + share_model_weights._weights) % self.Q