import re
import unittest
from bisect import bisect_right
from typing import List, Tuple

import pytz
from datetime import datetime


def now(fmt='%Y%m%d%H%M%S'):
    if fmt:
        return datetime.now().strftime(fmt)
    return datetime.now()


def format_datetime_as_iso_8601(dt: datetime):
    """
    :param dt:
    :return:
    """
    return dt.isoformat()


def append_timezone(dt: datetime, timezone_str: str, is_dst: bool = None):
    """
    Append timezone to the datetime object
    :param dt:
    :param timezone_str:
    :param is_dst: is it daylight saving time
    :return:
    """
    timezone = pytz.timezone(timezone_str)
    dt_aware = timezone.localize(dt, is_dst=is_dst)
    return dt_aware

def ensure_timezone(dt: datetime, timezone_str: str, is_dst: bool = None):
    if dt.tzinfo is None:
        return append_timezone(dt, timezone_str, is_dst)
    else:
        return dt.astimezone(pytz.timezone(timezone_str))

def parse_datetime_to_timestamp(dt: datetime, precision: str = 'ms') -> int:
    """
    Parse the datetime object to timestamp
    :param dt:
    :param precision:
    :return:
    """
    ratio = 1
    if precision == 's':
        ratio = 1
    elif precision == 'ms':
        ratio = 1_000
    elif precision == 'us':
        ratio = 1_000_000
    elif precision == 'ns':
        ratio = 1_000_000_000
    else:
        raise ValueError(f"Invalid precision: {precision}")
    
    return int(dt.timestamp() * ratio)

class StartEndLogTimeProcessor:
    @staticmethod
    def extract_start_end_time_from_log(content: str) -> List[Tuple[int, int]]:
        """
        Extract the start and end time from the log file
        :param content:
        :return:
        """
        start_pattern = r'Start time:\s*(\d+)'
        end_pattern = r'End time:\s*(\d+)'

        # Find all matches for start and end times
        start_times = re.findall(start_pattern, content)
        end_times = re.findall(end_pattern, content)

        return zip(start_times, end_times)

    @staticmethod
    def get_start_end_time_from_log_in_utc_ts(content: str) -> List[Tuple[int, int]]:
        """
        NOTE: This method is recommended because it uses UTC to combat the issue that timezone can change during the measurement
        Get the start and end time from the log file
        :param log:
        :return:
        """
        start_end_time_pairs = StartEndLogTimeProcessor.extract_start_end_time_from_log(content)
        # Convert timestamps to datetime objects and pair them
        timestamp_pairs = list(map(lambda x: (int(x[0]), int(x[1])), start_end_time_pairs))
        return timestamp_pairs
    
    @staticmethod
    def get_start_end_time_from_log(content: str, timezone_str: str = 'UTC') -> List[Tuple[datetime, datetime]]:
        """
        DEPRECATED: Use get_start_end_time_from_log_in_utc_ts() instead.
        NOTE: This method is deprecated because the timezone can change during the measurement.
        Get the start and end time from the log file
        :param log:
        :return:
        """
        start_end_time_pairs = StartEndLogTimeProcessor.extract_start_end_time_from_log(content)

        res = []
        for start, end in start_end_time_pairs:
            # Assume the timestamp is 13 digits (milliseconds)
            start_dt = datetime.fromtimestamp(int(start) / 1000.0, pytz.timezone(timezone_str))
            end_dt = datetime.fromtimestamp(int(end) / 1000.0, pytz.timezone(timezone_str))
            res.append((start_dt, end_dt))

        return res


class TimeIntervalQuery:
    def __init__(self, ts_traces: List[float]):
        self.ts_traces = ts_traces

    def query_interval_start_end_index(self, ts: float) -> (float, float):
        """
        Query the interval with start and end time that the timestamp ts belongs to
        :param ts:
        :return:
        """
        # bisect_right returns the insertion point for ts to maintain sorted order
        # if ts is in the list, bisect_right returns the index of the next element
        # if ts is not in the list, bisect_right returns the insertion index
        pos = bisect_right(self.ts_traces, ts)
        if pos == 0:
            return None, 0
        if pos == len(self.ts_traces):
            return len(self.ts_traces) - 1, None
        return pos - 1, pos

class TimezoneOffsetMinute:
    def __init__(self, offset_minutes: int):
        self.offset_minutes = offset_minutes
    
    @property
    def hours(self) -> int:
        return self.offset_minutes // 60
    
    @property
    def minutes(self) -> int:
        return abs(self.offset_minutes % 60)
    
    def __str__(self) -> str:
        sign = '-' if self.offset_minutes < 0 else '+'
        return f"{sign}{abs(self.hours):02d}:{self.minutes:02d}"

class TimezoneOffsetConverter:
    @staticmethod
    def parse_iso_offset_to_minute(offset_str: str) -> int:
        """Parse ISO 8601 offset string to minutes"""
        sign = -1 if offset_str[0] == '-' else 1
        hours = int(offset_str[1:3])
        minutes = int(offset_str[4:6]) if len(offset_str) > 5 else 0
        return sign * (TimezoneOffsetConverter.from_hour_to_minute(hours) + minutes)

    @staticmethod
    def parse_iso_offset_to_hour(offset_str: str) -> int:
        """Parse ISO 8601 offset string to minutes"""
        sign = -1 if offset_str[0] == '-' else 1
        hours = int(offset_str[1:3])
        minutes = int(offset_str[4:6]) if len(offset_str) > 5 else 0
        return sign * (hours + TimezoneOffsetConverter.from_minute_to_hour(minutes))
    
    @staticmethod
    def from_hour_to_minute(hours: float) -> float:
        return hours * 60
    
    @staticmethod
    def from_minute_to_hour(minutes: float) -> float:
        return minutes / 60

    @staticmethod
    def from_minute_to_second(minutes: float) -> float:
        return minutes * 60
    
    @staticmethod
    def from_second_to_minute(seconds: float) -> float:
        return seconds / 60

    @staticmethod
    def from_hour_to_second(hours: float) -> float:
        return hours * 3600
    
    @staticmethod
    def from_second_to_hour(seconds: float) -> float:
        return seconds / 3600
        
class TimestampUtil:
    @staticmethod
    def get_timestamp_unit(unix_ts: int) -> str:
        # Check from largest to smallest unit to avoid false positives
        if unix_ts >= 1e18:  # Nanoseconds (e.g. 1672531200000000000)
            return 'ns'
        elif unix_ts >= 1e15:  # Microseconds (e.g. 1672531200000000)
            return 'us'  
        elif unix_ts >= 1e12:  # Milliseconds (e.g. 1672531200000)
            return 'ms'
        else:  # Seconds (e.g. 1672531200)
            return 's'
    
    @staticmethod
    def convert_unix_ts_to_seconds(unix_ts: int) -> float:
        """Convert a unix timestamp to seconds, handling different time units.
        
        Args:
            unix_ts: Unix timestamp in ns, us, ms or s
            
        Returns:
            Timestamp converted to seconds as a float
        """
        unit = TimestampUtil.get_timestamp_unit(unix_ts)
        conversion_factors = {
            'ns': 1e9,
            'us': 1e6, 
            'ms': 1e3,
            's': 1
        }
        conversion_factor = conversion_factors.get(unit, None)
        if conversion_factor is None:
            raise ValueError(f"Invalid timestamp unit: {unit}")
        return unix_ts / conversion_factor
        
    @staticmethod
    def diff(ts1: int, ts2: int, unit: str = 's') -> float:
        conversion_factors = {
            'ns': 1e9,
            'us': 1e6, 
            'ms': 1e3,
            's': 1
        }
        ts1_seconds = TimestampUtil.convert_unix_ts_to_seconds(ts1)
        ts2_seconds = TimestampUtil.convert_unix_ts_to_seconds(ts2)
        ratio = conversion_factors.get(unit, None)
        if ratio is None:
            raise ValueError(f"Invalid timestamp unit: {unit}")
        return (ts1_seconds - ts2_seconds) * ratio
    

class LogTimeParser:
    @staticmethod
    def parse_time(dt: str, timezone_str: str | None) -> Tuple[datetime, int]:
        try:
            # Try parsing with offset, ignore timezone
            dt = datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S.%f%z")
            utc_ts = parse_datetime_to_timestamp(dt, precision='ms')
            utc_offset = dt.strftime('%z')
            utc_offset_minute = TimezoneOffsetConverter.parse_iso_offset_to_minute(utc_offset)
            return {
                'local_dt': dt.isoformat(timespec='microseconds'),
                'utc_ts': utc_ts,
                'utc_offset_minute': utc_offset_minute,
            }
        except ValueError:
            # Try parsing without offset, use the provided timezone
            dt = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S.%f")
            utc_offset_minute = 0
            if timezone_str:
                dt = append_timezone(dt, timezone_str)
                utc_offset = dt.strftime('%z')
                utc_offset_minute = TimezoneOffsetConverter.parse_iso_offset_to_minute(utc_offset)
            else:
                # Assume the timestamp is in UTC
                dt = append_timezone(dt, 'UTC')
            utc_ts = parse_datetime_to_timestamp(dt, precision='ms')
            return {
                'local_dt': dt.isoformat(timespec='microseconds'),
                'utc_ts': utc_ts,
                'utc_offset_minute': utc_offset_minute,
            }

class Unittest(unittest.TestCase):
    def test_start_end_time_extraction(self):
        utc_timezone = pytz.timezone('UTC')
        d1 = datetime(2021, 6, 21, 0, 0, 0).astimezone(utc_timezone)
        d2 = datetime(2021, 6, 21, 0, 0, 1).astimezone(utc_timezone)
        d3 = datetime(2021, 6, 21, 0, 0, 2).astimezone(utc_timezone)
        d4 = datetime(2021, 6, 21, 0, 0, 3).astimezone(utc_timezone)
        content = f"""
        Start time: {int(d1.timestamp() * 1000)} 
        End time: {int(d2.timestamp() * 1000)}
        Start time: {int(d3.timestamp() * 1000)}
        End time: {int(d4.timestamp() * 1000)}
        """
        time_pairs = StartEndLogTimeProcessor.get_start_end_time_from_log(content)
        self.assertEqual(len(time_pairs), 2)
        self.assertEqual(time_pairs[0][0], d1)
        self.assertEqual(time_pairs[0][1], d2)
        self.assertEqual(time_pairs[1][0], d3)
        self.assertEqual(time_pairs[1][1], d4)