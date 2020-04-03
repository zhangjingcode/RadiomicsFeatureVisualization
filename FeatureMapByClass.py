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
from FeatureMapShow import FeatureMapVisualizition


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

    def seek_single_candidate_case(self, feature_name, case_num):
        sub_feature_pd = self.feature_pd[['label', feature_name]].copy()
        sub_feature_pd.sort_values(by=feature_name, inplace=True)
        sorted_index_list = sub_feature_pd.axes[0]
        max_case, min_case = sorted_index_list[-1], sorted_index_list[0]

        max_info = max_case + '(' + str(int(sub_feature_pd.at[max_case, 'label'])) + ')'
        min_info = min_case + '(' + str(int(sub_feature_pd.at[min_case, 'label'])) + ')'
        print('{} value maximum case : {}, minimum case : {}'.format(feature_name, max_info, min_info))
        top_case_list = list(sorted_index_list)[-case_num:]
        last_case_list = list(sorted_index_list)[:case_num]
        return top_case_list, last_case_list

    def seek_candidate_case(self, feature_csv_path, selected_feature_list, case_num):
        """
            seek a candidate image by feature value from feature csv， print the candidate case.

        Parameters
        ----------
        feature_csv_path : str, radiomics feature csv;
        selected_feature_list : list, selected feature name list;


        """
        self.load(feature_csv_path, selected_feature_list)
        candidate_case_list = []
        candidate_case_dict = {}
        for sub_feature in selected_feature_list:
            top_case_list, last_case_list = self.seek_single_candidate_case(sub_feature, case_num)
            candidate_case_dict[sub_feature] = {'top': top_case_list, 'last': last_case_list}

        # seek common case
        for sub_feature in list(candidate_case_dict.keys()):
            all_features = candidate_case_dict[sub_feature]['top'] + candidate_case_dict[sub_feature]['last']
            if len(candidate_case_list) == 0:
                candidate_case_list = all_features

            else:
                candidate_case_list = list(set(candidate_case_list).intersection(set(all_features)))

        # check common case

        for sub_feature in list(candidate_case_dict.keys()):
            sub_checked_case = list(set(candidate_case_dict[sub_feature]['top']).
                                    intersection(set(candidate_case_list)))

            candidate_case_dict[sub_feature]['top'] = [index + "(" + str(int(self.feature_pd.at[index, 'label'])) + ")"
                                                       for index in sub_checked_case]

            sub_checked_case = list(set(candidate_case_dict[sub_feature]['last']).
                                    intersection(set(candidate_case_list)))
            candidate_case_dict[sub_feature]['last'] = [index + "(" + str(int(self.feature_pd.at[index, 'label'])) + ")"
                                                        for index in sub_checked_case]

        df = pd.DataFrame.from_dict(candidate_case_dict, orient='index')
        print(df)

    @staticmethod
    def decode_feature_name(feature_name_list):
        sub_filter_name = ''
        img_setting = {'imageType': 'Original'}
        feature_dict = {}
        for sub_feature in feature_name_list:

            # big feature class
            if sub_feature in ['firstorder', 'glcm', 'glrlm', 'ngtdm', 'glszm']:
                sub_feature_setting = {sub_feature: []}
                feature_dict.update(sub_feature_setting)

            else:
                img_type = sub_feature.split('_')[-3]
                if img_type.rfind('wavelet') != -1:

                # if img_type in ['LLL', 'HLL','LHL', 'LLH', 'HHL', 'HHH','HLH','LHH']:
                    img_setting['imageType'] = 'Wavelet'
                    sub_filter_name = img_type.split('-')[-1]
                elif img_type.rfind('LOG') != -1:
                    img_setting['imageType'] = 'LoG'
                    sub_filter_name = img_type

                else:
                    img_setting['imageType'] = img_type
                # if img_type not in img_setting['imageType']:
                #     img_setting['imageType'].append(img_type)

                feature_class = sub_feature.split('_')[-2]
                feature_name = sub_feature.split('_')[-1]

                if feature_class not in feature_dict.keys():
                    feature_dict[feature_class] = []
                    feature_dict[feature_class].append(feature_name)
                else:
                    feature_dict[feature_class].append(feature_name)
        print(img_setting)
        print(feature_dict)
        return img_setting, feature_dict, sub_filter_name

    # crop img by kernelRadius, remove redundancy slice to speed up,
    def crop_img(self, original_roi_path, original_img_path, store_key=''):
        roi = sitk.ReadImage(original_roi_path)

        roi_array = sitk.GetArrayFromImage(roi)
        max_roi_slice_index = np.argmax(np.sum(roi_array, axis=(1, 2)))

        z_range = [max_roi_slice_index - self.kernelRadius, max_roi_slice_index + self.kernelRadius + 1]
        x_index = np.where(np.sum(roi_array[max_roi_slice_index], axis=0) > 0)[0]
        x_range = [min(x_index) - self.kernelRadius, max(x_index) + self.kernelRadius + 1]
        y_index = np.where(np.sum(roi_array[max_roi_slice_index], axis=1) > 0)[0]
        y_range = [min(y_index) - self.kernelRadius, max(y_index) + self.kernelRadius + 1]


        cropped_roi_array = roi_array[z_range[0]:z_range[1]]
        cropped_roi = sitk.GetImageFromArray(cropped_roi_array)
        cropped_roi.SetDirection(roi.GetDirection())
        cropped_roi.SetOrigin(roi.GetOrigin())
        cropped_roi.SetSpacing(roi.GetSpacing())

        img = sitk.ReadImage(original_img_path)
        img_array = sitk.GetArrayFromImage(img)
        cropped_img_array = img_array[z_range[0]:z_range[1]]
        cropped_img = sitk.GetImageFromArray(cropped_img_array)
        cropped_img.SetDirection(img.GetDirection())
        cropped_img.SetOrigin(img.GetOrigin())
        cropped_img.SetSpacing(img.GetSpacing())

        roi_info = [roi.GetDirection(), roi.GetOrigin(), roi.GetSpacing()]
        img_info = [img.GetDirection(), img.GetOrigin(), img.GetSpacing()]
        index_dict = {0:'direction', 1:'origin', 2:'spacing'}
        start = 0
        for sub_roi_info, sub_img_info in zip(roi_info, img_info):
            if sub_roi_info != sub_img_info:
                print(index_dict[start], 'failed')
                print('roi:', sub_roi_info)
                print('img:', sub_img_info)
        # check cropped image

        from MeDIT.Visualization import Imshow3DArray
        from MeDIT.Normalize import Normalize01
        # Imshow3DArray(Normalize01(np.transpose(cropped_img_array, (1,2,0))),
        #               roi=np.transpose(cropped_roi_array, (1,2,0)))
        sitk.WriteImage(cropped_img, os.path.join(self.store_path, store_key + '_cropped_img.nii.gz'))
        sitk.WriteImage(cropped_roi, os.path.join(self.store_path, store_key + '_cropped_roi.nii.gz'))
        self.sub_img_array = np.transpose(cropped_img_array, (1, 2, 0))
        self.sub_roi_array = np.transpose(cropped_roi_array, (1, 2, 0))
        return cropped_img, cropped_roi

    def generate_feature_map(self, candidate_img_path, candidate_roi_path, kernelRadius, feature_name_list, store_path):
        """
            Generate specific feature map based on kernel Radius.

        Parameters
        ----------
        candidate_img_path: str, candidate image path;
        candidate_roi_path: str, candidate ROI path;
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
        parameter_path = r'D:\MyScript\RadiomicsVisualization\RadiomicsFeatureVisualization\RadiomicsParams.yaml'
        setting_dict = {'label': 1, 'interpolator': 'sitkBSpline', 'correctMask': True,
                        'geometryTolerance': 1, 'kernelRadius': self.kernelRadius,
                        'maskedKernel': True, 'voxelBatch': 50}

        extractor = featureextractor.RadiomicsFeaturesExtractor(parameter_path, self.store_path)
        extractor.disableAllImageTypes()
        extractor.disableAllFeatures()

        img_setting, feature_dict, sub_filter_name = self.decode_feature_name(feature_name_list)
        extractor.enableImageTypeByName(**img_setting)
        extractor.enableFeaturesByName(**feature_dict)

        cropped_original_img, cropped_original_roi = self.crop_img(candidate_roi_path, candidate_img_path,
                                                                   store_key='original')

        if sub_filter_name:
            # generate filter image
            extractor.execute(candidate_img_path, candidate_roi_path, voxelBased=False)
            candidate_img_path = os.path.join(self.store_path, sub_filter_name+'.nii.gz')
            cropped_filter_img, cropped_filter_roi = self.crop_img(candidate_roi_path, candidate_img_path,
                                                                   store_key=sub_filter_name)
            result = extractor.execute(cropped_filter_img, cropped_filter_roi, voxelBased=True)
        #
        #
        else:
            result = extractor.execute(cropped_original_img, cropped_original_roi , voxelBased=True)
        # without parameters, glcm ,kr=5 ,646s ,cropped img, map shape (5, 132, 128)
        # without parameters, glcm ,kr=1 ,386s ,cropped img, map shape (3, 122, 128)

        # without parameters, glcm ,kr=1 ,566s ,without cropped img, map shape (5, 132, 128)

        # extract original image
        # result = extractor.execute(candidate_img_path, candidate_roi_path, voxelBased=True)
        for key, val in six.iteritems(result):
            if isinstance(val, sitk.Image):
                shape = (sitk.GetArrayFromImage(val)).shape
                print('feature_map shape is ', shape)
                # Feature map
                sitk.WriteImage(val, store_path + '\\' + key + '.nrrd', True)
                print(time.time() - start_time)

    def show_feature_map(self, show_img_path, show_roi_path, show_feature_map_path, store_path):
        from MeDIT.SaveAndLoad import LoadNiiData
        _, _, feature_map_array = LoadNiiData(show_feature_map_path)
        featuremapvisualization = FeatureMapVisualizition()
        featuremapvisualization.LoadData(show_img_path, show_roi_path, show_feature_map_path)

        # hsv/jet/gist_rainbow
        featuremapvisualization.Show(color_map='rainbow', store_path=store_path)


def main():
    feature_mapper = FeatureMapper()
    t1c_features_list = ['original_glcm_ClusterProminence', 'original_glcm_Imc1', 'original_glcm_Imc2',
                         'original_glcm_MCC', 'original_shape_Sphericity', 'original_shape_SurfaceVolumeRatio']

    t2_features_list = ['original_glcm_DifferenceEntropy', 'original_glrlm_LongRunEmphasis',
                        'original_glrlm_RunVariance', 'original_glszm_SizeZoneNonUniformityNormalized',
                        'original_glszm_HighGrayLevelZoneEmphasis',
                        'original_glszm_SizeZoneNonUniformityNormalized',
                        'original_glszm_SmallAreaEmphasis', 'original_glszm_SmallAreaHighGrayLevelEmphasis']

    selected_features_list = ['T1C_' + index for index in t1c_features_list] + ['T2_' + index for index in t2_features_list]
    feature_mapper.seek_candidate_case(r'D:\hospital\EENT\New_test\SeparateByDate\3D_3.0\features3D.csv',
                                       selected_features_list, 20)
    img_1_path = r'D:\hospital\EENT\code\FeatureMap\N45\data1.nii'
    roi_1_path = r'D:\hospital\EENT\code\FeatureMap\N45\ROI.nii'
    store_1_path = r'D:\hospital\EENT\code\FeatureMap\N45_1_T2_feature_map'
    #
    img_0_path = r'D:\hospital\EENT\code\FeatureMap\N28\data1.nii'
    roi_0_path = r'D:\hospital\EENT\code\FeatureMap\N28\ROI.nii'
    store_0_path = r'D:\hospital\EENT\code\FeatureMap\N28_0_T2_feature_map'
    # data1 T2, data3 T1C Reg

    # feature_mapper.generate_feature_map(img_1_path, roi_1_path, 1, t2_features_list,
    #                                     r'D:\hospital\EENT\code\FeatureMap\N45_1_T2_feature_map')
    # feature_mapper.generate_feature_map(img_0_path, roi_0_path, 1, t2_features_list,
    #                                     r'D:\hospital\EENT\code\FeatureMap\N28_0_T2_feature_map')
    # for sub_feature in t2_features_list:
    # #
    #     feature_map_path = os.path.join(store_0_path, sub_feature+'.nrrd')
    #     feature_mapper.show_feature_map(img_0_path, roi_0_path, feature_map_path, os.path.join(store_0_path, sub_feature))
