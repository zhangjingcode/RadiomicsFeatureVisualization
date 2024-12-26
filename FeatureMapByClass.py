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
from pathlib import Path

from SKMRradiomics import featureextractor
# from radiomics import featureextractor
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
                # extract all features
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
                    img_setting['imageType'] = 'Original'
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

        sitk.WriteImage(cropped_img, os.path.join(self.store_path, store_key + '_cropped_img.nii.gz'))
        sitk.WriteImage(cropped_roi, os.path.join(self.store_path, store_key + '_cropped_roi.nii.gz'))
        self.sub_img_array = np.transpose(cropped_img_array, (1, 2, 0))
        self.sub_roi_array = np.transpose(cropped_roi_array, (1, 2, 0))
        print('ROI size: ', np.sum(cropped_roi))
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
        feature_name_list: [str], [feature_name1, feature_name2,...] or ['glcm', 'glrlm']
        store_path: str;

        Returns
        -------

        """

        start_time = time.time()
        self.kernelRadius = kernelRadius
        self.store_path = store_path
        parameter_path = r'D:\MyScript\RadiomicsVisualization\RadiomicsFeatureVisualization\RadiomicsParams.yaml'
        setting_dict = {'label': 1, 'interpolator': 'sitkBSpline', 'correctMask': True,
                        'geometryTolerance': 10, 'kernelRadius': self.kernelRadius,
                        'maskedKernel': True, 'voxelBatch': 50}

        extractor = featureextractor.RadiomicsFeaturesExtractor(parameter_path, self.store_path, **setting_dict)
        extractor.disableAllImageTypes()
        extractor.disableAllFeatures()

        img_setting, feature_dict, sub_filter_name = self.decode_feature_name(feature_name_list)
        extractor.enableImageTypeByName(**img_setting)
        extractor.enableFeaturesByName(**feature_dict)

        cropped_original_img, cropped_original_roi = self.crop_img(candidate_roi_path, candidate_img_path,
                                                                   store_key='original')

        if sub_filter_name:
            # generate filter image firstly for speeding up
            extractor.execute(candidate_img_path, candidate_roi_path, voxelBased=False)
            candidate_img_path = os.path.join(self.store_path, sub_filter_name+'.nii.gz')
            cropped_filter_img, cropped_filter_roi = self.crop_img(candidate_roi_path, candidate_img_path,
                                                                   store_key=sub_filter_name)
            result = extractor.execute(cropped_filter_img, cropped_filter_roi, voxelBased=True)
        #
        #
        else:
            result = extractor.execute(cropped_original_img, cropped_original_roi, voxelBased=True)
        # without parameters, glcm ,kr=5 ,646s ,cropped img, map shape (5, 132, 128)
        # without parameters, glcm ,kr=1 ,386s ,cropped img, map shape (3, 122, 128)

        # without parameters, glcm ,kr=1 ,566s ,without cropped img, map shape (5, 132, 128)

        # extract original image

        for key, val in six.iteritems(result):
            if isinstance(val, sitk.Image):
                shape = (sitk.GetArrayFromImage(val)).shape
                # Feature map
                sitk.WriteImage(val, os.path.join(store_path, key + '.nrrd'), True)


    def show_feature_map(self, show_img_path, show_roi_path, show_feature_map_path, store_path):
        feature_map_img = sitk.ReadImage(show_feature_map_path)
        feature_map_array = sitk.GetArrayFromImage(feature_map_img)
        feature_map_array.transpose(1, 2, 0)
        feature_map_visualization = FeatureMapVisualizition()
        feature_map_visualization.LoadData(show_img_path, show_roi_path, show_feature_map_path)

        # hsv/jet/gist_rainbow
        feature_map_visualization.Show(color_map='rainbow', store_path=store_path)


def main():
    feature_mapper = FeatureMapper()
    features_name_list = ['original_glcm_DifferenceEntropy', 'original_glrlm_LongRunEmphasis',
                        'original_glrlm_RunVariance', 'original_glszm_SizeZoneNonUniformityNormalized']

    features_class_list = ['glcm', 'glrlm']
    cur_file_path = Path(__file__).absolute().parent
    img_path = cur_file_path / 'BreastRADERData' / 'DCE.nii.gz'
    roi_path = cur_file_path / 'BreastRADERData' / 'DCE_ROI.nii.gz'

    store_path = cur_file_path / 'BreastRADERData' / 'FeatureMap'

    if not Path(store_path).exists():
        Path(store_path).mkdir()

    feature_mapper.generate_feature_map(str(img_path), str(roi_path), 1, features_name_list, str(store_path))

    cropped_img_path = cur_file_path / 'BreastRADERData' / 'FeatureMap' / 'original_cropped_img.nii.gz'
    cropped_roi_path = cur_file_path / 'BreastRADERData' / 'FeatureMap' / 'original_cropped_roi.nii.gz'
    feature_name = 'original_glszm_SizeZoneNonUniformityNormalized'
    feature_map = cur_file_path / 'BreastRADERData' / 'FeatureMap' / str(feature_name+'.nrrd')
    fig_save_path = cur_file_path / 'BreastRADERData' / 'FeatureMap' / feature_name
    feature_mapper.show_feature_map(str(cropped_img_path), str(cropped_roi_path), str(feature_map),
                                    str(fig_save_path))

if __name__ == '__main__':
    main()
