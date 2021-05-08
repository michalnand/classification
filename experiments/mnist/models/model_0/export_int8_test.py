import sys
sys.path.insert(0,'../../../..')

import libs
import libs_dataset

import model         as ModelReference
import model_testing_int8 as ModelTesting

dataset = libs_dataset.DatasetMnist()

libs.ExportTest(dataset, 64, ModelReference, ModelTesting)