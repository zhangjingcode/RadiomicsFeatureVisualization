#!/D:/anaconda python
# -*- coding: utf-8 -*-
# @Project : MyScript
# @FileName: FeatureMapByClass.py
# @IDE: PyCharm
# @Time  : 2020/3/9 21:16
# @Author : Jing.Z
# @Email : zhangjingmri@gmail.com
# @Desc : ==============================================
# Life is Short I Use Python!!!                      ===
# ======================================================


import os
import time
import six

import SimpleITK as sitk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SKMRradiomics import featureextractor

from RadiomicsFeatureVisualization.FeatureMapShow import FeatureMapVisualizition

class FeatureMapper:
    """

        This class would found a candidate case image to generate radiomics feature map based voxel. /

    """

    def __init__(self):
        self.feature_pd = pd.DataFrame()
        self.selected_feature_list = []
        self.store_path = ''
        self.kernelRadius = ''
        self.sub_img_array = np.array([])
        self.sub_roi_array = np.array([])

    def load(self, feature_csv_path, selected_feature_list):
        self.feature_pd = pd.read_csv(feature_csv_path, index_col=0)
        self.selected_feature_list = selected_feature_list

    def seek_single_candidate_case(self, feature_name):
        sub_feature_pd = self.feature_pd[['label', feature_name]].copy()
        sub_feature_pd.sort_values(by=feature_name, inplace=True)
        sorted_index_list = sub_feature_pd.axes[0]
        max_case, min_case = sorted_index_list[-1], sorted_index_list[0]

        max_info = max_case + '(' + str(int(sub_feature_pd.at[max_case, 'label'])) + ')'
        min_info = min_case + '(' + str(int(sub_feature_pd.at[min_case, 'label'])) + ')'
        print('{} value maximum case : {}, minimum case : {}'.format(feature_name,max_info, min_info))

    def seek_candidate_case(self, feature_csv_path, selected_feature_list):
        """
            seek a candidate image by feature value from feature csv， print the candidate case.

        Parameters
        ----------
        feature_csv_path : str, radiomics feature csv;
        selected_feature_list : list, selected feature name list;


        """
        self.load(feature_csv_path, selected_feature_list)

        for sub_feature in selected_feature_list:
            self.seek_single_candidate_case(sub_feature)

    @staticmethod
    def decode_feature_name(feature_name):
        img_type = feature_name.split('_')[-3]
        img_setting = {'imageType': img_type}

        feature_class = feature_name.split('_')[-2]
        feature_name = feature_name.split('_')[-1]
        feature_dict = {feature_class: [feature_name]}

        return img_setting, feature_dict

    # crop img by kernelRadius, remove redundancy slice to speed up
    def crop_img(self, roi_path, img_path):
        roi = sitk.ReadImage(roi_path)

        roi_array = sitk.GetArrayFromImage(roi)
        max_roi_slice_index = np.argmax(np.sum(roi_array, axis=(1, 2)))

        z_range = [max_roi_slice_index-self.kernelRadius, max_roi_slice_index+self.kernelRadius+1]
        x_index = np.where(np.sum(roi_array[max_roi_slice_index], axis=0) > 0)[0]
        x_range = [min(x_index) - self.kernelRadius, max(x_index) + self.kernelRadius + 1]
        y_index = np.where(np.sum(roi_array[max_roi_slice_index], axis=1) > 0)[0]
        y_range = [min(y_index) - self.kernelRadius, max(y_index) + self.kernelRadius + 1]

        cropped_roi_array = roi_array[z_range[0]:z_range[1]]
        cropped_roi = sitk.GetImageFromArray(cropped_roi_array)
        cropped_roi.SetDirection(roi.GetDirection())
        cropped_roi.SetOrigin(roi.GetOrigin())
        cropped_roi.SetSpacing(roi.GetSpacing())

        img = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage(img)
        cropped_img_array = img_array[z_range[0]:z_range[1]]
        cropped_img = sitk.GetImageFromArray(cropped_img_array)
        cropped_img.SetDirection(img.GetDirection())
        cropped_img.SetOrigin(img.GetOrigin())
        cropped_img.SetSpacing(img.GetSpacing())

        from MeDIT.Visualization import Imshow3DArray
        from MeDIT.Normalize import Normalize01
        Imshow3DArray(Normalize01(np.transpose(cropped_img_array, (1,2,0))),
                      roi=np.transpose(cropped_roi_array, (1,2,0)))
        sitk.WriteImage(cropped_img, os.path.join(self.store_path, 'cropped_img.nii.gz'))
        sitk.WriteImage(cropped_roi, os.path.join(self.store_path, 'cropped_roi.nii.gz'))
        self.sub_img_array = np.transpose(cropped_img_array, (1, 2, 0))
        self.sub_roi_array = np.transpose(cropped_roi_array, (1, 2, 0))
        return cropped_img, cropped_roi

    def generate_feature_map(self, img_path, roi_path, kernelRadius, feature_name, store_path):
        """
            Generate specific feature map based on kernel Radius.

        Parameters
        ----------
        img_path: str, candidate image path;
        roi_path: str, candidate ROI path;
        kernelRadius: integer, specifies the size of the kernel to use as the radius from the center voxel. \
                    Therefore the actual size is 2 * kernelRadius + 1. E.g. a value of 1 yields a 3x3x3 kernel, \
                    a value of 2 5x5x5, etc. In case of 2D extraction, the generated kernel will also be a 2D shape
                    (square instead of cube).
        feature_name: str;
        store_path: str;

        Returns
        -------

        """

        start_time = time.time()
        self.kernelRadius = kernelRadius
        self.store_path = store_path
        parameter_path = r'D:\hospital\EENT\code\FeatureMap\RadiomicsParams.yaml'
        setting_dict = {'label': 1, 'interpolator': 'sitkBSpline', 'correctMask': True,
                        'geometryTolerance': 0.1, 'kernelRadius': self.kernelRadius,
                        'maskedKernel': True, 'voxelBatch':50}

        extractor = featureextractor.RadiomicsFeaturesExtractor(**setting_dict)
        extractor.disableAllFeatures()

        img_setting, feature_dict = self.decode_feature_name(feature_name)

        extractor.enableFeaturesByName(**img_setting, **feature_dict)

        cropped_img, cropped_roi = self.crop_img(roi_path, img_path)
        result = extractor.execute(cropped_img, cropped_roi, voxelBased=True)
        # without parameters, glcm ,kr=5 ,646s ,cropped img, map shape (5, 132, 128)
        # without parameters, glcm ,kr=1 ,386s ,cropped img, map shape (3, 122, 128)

        # without parameters, glcm ,kr=1 ,566s ,without cropped img, map shape (5, 132, 128)
        # result = extractor.execute(img_path, roi_path, voxelBased=True)
        for key, val in six.iteritems(result):
            if isinstance(val, sitk.Image):
                shape = (sitk.GetArrayFromImage(val)).shape
                print('feature_map shape is ', shape)
                # Feature map
                sitk.WriteImage(val, store_path + '\\' + key + '.nrrd', True)
                print(time.time()-start_time)

    def show_feature_map(self, img_path, roi_path, feature_map_path):
        from MeDIT.SaveAndLoad import LoadNiiData
        _, _, feature_map_array = LoadNiiData(feature_map_path)
        featuremapvisualization = FeatureMapVisualizition()
        featuremapvisualization.LoadData(r'D:\hospital\EENT\code\FeatureMap\N45\data1.nii',
                                         r'D:\hospital\EENT\code\FeatureMap\N45\ROI.nii', feature_map_path)

        store_figure_path = self.store_path + '\\' + (os.path.split(feature_map_path)[-1]).split('.')[0]
        # hsv/jet/gist_rainbow
        featuremapvisualization.Show(color_map='rainbow',store_path=store_figure_path)


#
feature_mapper = FeatureMapper()
# #
# feature_mapper.seek_candidate_case(r'D:\hospital\EENT\New_test\SeparateByDate\3D_3.0\features3D.csv',
#                     ['T2_original_glcm_DifferenceEntropy', 'T2_original_glrlm_LongRunEmphasis',
#                      'T2_original_glszm_HighGrayLevelZoneEmphasis'])
img_path = r'D:\hospital\EENT\code\FeatureMap\N45\data1.nii'
roi_path = r'D:\hospital\EENT\code\FeatureMap\N45\ROI.nii'

feature_mapper.generate_feature_map(img_path, roi_path, 1,
                                    'original_glcm_DifferenceEntropy',
                                    r'D:\hospital\EENT\code\FeatureMap\feature_map')

feature_map_path =r'D:\hospital\EENT\code\FeatureMap\feature_map\original_glcm_DifferenceEntropy.nrrd'
feature_mapper.show_feature_map(img_path, roi_path, feature_map_path)
