import enum
import os

from scripts.constants import DATASET_DIR, OUTPUT_DIR as OUTPUT_DIR_ROOT

DATASET_NAME = 'la_to_omaha_road_test_202411'

ROOT_DIR = os.path.join(DATASET_DIR, DATASET_NAME)
OUTPUT_DIR = os.path.join(OUTPUT_DIR_ROOT, DATASET_NAME)

TIMEZONE_EASTERN = 'America/New_York' # UTC-5
TIMEZONE_CST = 'America/Chicago' # UTC-6
TIMEZONE_MOUNTAIN = 'America/Denver' # UTC-7
TIMEZONE_PACIFIC = 'America/Los_Angeles' # UTC-8

class DatasetLabel(enum.Enum):
    NORMAL = 'NORMAL'
    TEST_BOSTON_DATA = 'TEST_BOSTON_DATA'
    TEST_DATA = 'TEST_DATA'
    MALFORMED_DATA = 'MALFORMED_DATA'