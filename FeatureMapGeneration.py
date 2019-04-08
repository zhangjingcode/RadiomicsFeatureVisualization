from SKMRradiomics import featureextractor
# import os
import csv
import copy
import logging
import SimpleITK as sitk
import glob
import six


def GetFeatureMap(params_path, store_path, image_path, roi_path):
    extractor = featureextractor.RadiomicsFeaturesExtractor(params_path, store_path)
    result = extractor.execute(image_path, roi_path, voxelBased=False)
    print(result)
    # for key, val in six.iteritems(result):
    #     if isinstance(val, sitk.Image):
    #         shape = (sitk.GetArrayFromImage(val)).shape
    #         print('feature_map shape is ', shape)
    #         # Feature map
    #         sitk.WriteImage(val, store_path + '\\' + key + '.nrrd', True)
    #     else:  # Diagnostic information
    #         print("\t%s: %s" % (key, val))

if __name__ == "__main__":

    image_path = r'C:\Users\zj\Desktop\SHGH\feature_map\3 Sag T2.nii.gz'
    roi_path = r'C:\Users\zj\Desktop\SHGH\feature_map\Untitled.nii.gz'
    save_path = r'D:\MyScript\RadiomicsFeatureVisualization\demo2'
    GetFeatureMap(r'D:\MyScript\RadiomicsParams.yaml', save_path, image_path, roi_path)
