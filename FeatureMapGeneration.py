from SKMRradiomics import featureextractor as skmrradiomics
from radiomics import featureextractor as originalradiomcis
# import os
import csv
import copy
import logging
import SimpleITK as sitk
import glob
import six


def GetFeatureMap(params_path, store_path, image_path, roi_path,voxelBasedSet=True):
    if not voxelBasedSet:
        extractor = skmrradiomics.RadiomicsFeaturesExtractor(params_path, store_path)
        result = extractor.execute(image_path, roi_path, voxelBased=voxelBasedSet)
    if voxelBasedSet:
        extractor = originalradiomcis.RadiomicsFeaturesExtractor(params_path, store_path)
        result = extractor.execute(image_path, roi_path, voxelBased=voxelBasedSet)
        for key, val in six.iteritems(result):
            if isinstance(val, sitk.Image):
                shape = (sitk.GetArrayFromImage(val)).shape
                print('feature_map shape is ', shape)
                # Feature map
                sitk.WriteImage(val, store_path + '\\' + key + '.nrrd', True)
            else:  # Diagnostic information
                print("\t%s: %s" % (key, val))


if __name__ == "__main__":

    image_path = r'D:\MyScript\RadiomicsVisualization\RadiomicsFeatureVisualization\data2.nii.gz'
    roi_path = r'D:\MyScript\RadiomicsVisualization\RadiomicsFeatureVisualization\ROI.nii.gz'
    save_path = r'D:\MyScript\RadiomicsVisualization\RadiomicsFeatureVisualization'
    GetFeatureMap(r'D:\MyScript\RadiomicsParams.yaml', save_path, image_path, roi_path)
