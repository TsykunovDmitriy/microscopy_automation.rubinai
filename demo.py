import argparse
import os

import microscopy_automation

from microscopy_automation.utils.data import Data

from microscopy_automation.utils.preprocessing import get_preprocessing_for_segmentation, preprocessing_for_predict_mask, get_preprocessing_for_classification
from microscopy_automation.utils.functions import cutLeukocyte, getBoundingBox, colorBnbCls

from microscopy_automation.utils.model import SegmentationModel, ClassificationModel

import cv2

#-------------Parsing Arguments-------------#
parser = argparse.ArgumentParser()

parser.add_argument("-dd", "--DATA_DIR", 
                    help="write a directory containing white blood cell images", 
                    default='images/')

parser.add_argument("-rd", "--RESULTS_DIR", 
                    help="write a directory where will be save results of pipeline's work", 
                    default='results/')

args = parser.parse_args()

assert os.path.isdir(args.DATA_DIR), 'Uncorreced path to DATA_DIR directory'

if not(os.path.isdir(args.RESULTS_DIR)):
    os.mkdir(args.RESULTS_DIR)

def main(data_dir, results_dir):
    print('Loading models is...')
    segmentaton_model = SegmentationModel(path=microscopy_automation.PATH_TO_SEGMENTATION_MODEL)
    classification_model = ClassificationModel(path=microscopy_automation.PATH_TO_CLASSIFICATION_MODEL)
    print('Done!')
    transforms_for_segmentation = get_preprocessing_for_segmentation(segmentaton_model.get_preprocessing_fn())
    transforms_for_classification = get_preprocessing_for_classification()

    print('Loading data is...')
    data = Data(data_dir, results_dir, preprocessing_fn=transforms_for_segmentation)
    print('Done!')
    print('For stopping click Control-C')
    try:
        while True:
            for idx, image, inputs in data:
                print(f'Working with file {data.info["path_to_image"][idx]} is...')
                #predict mask
                pr_mask = segmentaton_model.predict(inputs)
                mask = preprocessing_for_predict_mask(pr_mask, image.shape, threshold=0.3)
                coor_bnb = getBoundingBox(image, mask)
                #classification
                for_classifier = cutLeukocyte(image, coor_bnb)
                cls_pred = []
                for inp_cls in for_classifier:
                    cls_pred.append(classification_model.predict(inp_cls, preprocessing_fn=transforms_for_classification))
                #color bounding boxes and write class on image
                colorBnbCls(image, coor_bnb, cls_pred)
                #save result
                path_to_result = '{}{}{}'.format(results_dir, 'result_', os.path.basename(data.info['path_to_image'][idx]))
                cv2.imwrite(path_to_result, image)
                data.writeInfo(idx=idx, path_to_result=path_to_result, classes=cls_pred, bn_boxes=coor_bnb, flag=True)
                print('Done!')
            data.updateInfo()
    except KeyboardInterrupt:
        print('Saving result...')
        data.save_to_json()
        print('Done! Bye:)')

if __name__ == "__main__":
    main(args.DATA_DIR, args.RESULTS_DIR)