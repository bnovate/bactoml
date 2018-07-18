#!/usr/bin/env python
# -*- coding: UTF8 -*-
"""
bactoml.test_template
=====================

Example tests performed on the scikit-learn tempalte

Tests follow the py.tests pattern.

:author: Douglas Watson <douglas.watson@bnovate.com>
:date: 2018
:license: Private code, see LICENSE for more information

"""

import numpy as np
import pytest

from sklearn.utils.estimator_checks import check_estimator

from bactoml.template import TemplateTransformer, TemplateEstimator

@pytest.fixture()
def array_data():
    return np.array([1, 4, 9]).reshape(-1, 1)  # needs vertical array


class TestTemplateTransformer(object):

    def test_fit_ok(self, array_data):
        """ Fit should work on valid data """
        tf = TemplateTransformer()
        tf.fit(array_data)

    def test_fit_invalid(self):
        """ Fit should fail on invalid data """
        tf = TemplateTransformer()
        wrong_labels = ['a', 'b']
        with pytest.raises(TypeError):
            tf.fit(array_data, wrong_labels)

    def test_transform(self, array_data):
        """ Transform should return the transformed data """
        tf = TemplateTransformer()
        tf.fit(array_data)
        transf = tf.transform(array_data)
        assert list(transf) == [1, 2, 3]


class TestEstimatorTemplate(object):

    def test_estimator_validity(self):
        """ TemplateEstimator needs to be a valid sklearn object """
        check_estimator(TemplateEstimator)
