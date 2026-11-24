from pathlib import Path

from utils import create_data_lists

if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent
    create_data_lists(voc07_path=project_dir / 'data/VOCdevkit3/VOC2007',
                      voc12_path=project_dir / 'data/VOCdevkit/VOC2012',
                      output_folder=project_dir,
                      voc07_test_path=project_dir / 'data/VOCdevkit2/VOC2007')
