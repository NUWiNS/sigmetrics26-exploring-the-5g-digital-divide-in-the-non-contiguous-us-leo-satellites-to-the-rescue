
from bisect import bisect_right
from datetime import datetime
from typing import Any, List, Tuple

 
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

class TypeIntervalQueryUtil:
    def __init__(self, data: List[Tuple[int, Any]], coherent_time_sec: float = None):
        """
        :param data: List of tuples containing utc timestamp and value, e.g. [(utc_ts, value)]
        :param coherent_time_ms: Maximum allowed time difference in milliseconds, None by default
        """
        self.data = data
        self.coherent_time_sec = coherent_time_sec
        self.ts_traces: List[float] = []
        self.interval_query: TimeIntervalQuery | None = None

    def build_interval_query(self):
        self.ts_traces = [utc_ts for utc_ts, _ in self.data]
        self.interval_query = TimeIntervalQuery(self.ts_traces)

    def query(self, ts: datetime | float) -> Any | None:
        """
        Query the value at the timestamp ts
        :param ts: timestamp to query (datetime or float)
        :return: value if found within threshold, None otherwise
        """
        ts = float(ts.timestamp()) if isinstance(ts, datetime) else ts
        if not self.interval_query:
            self.build_interval_query()
        start_i, end_i = self.interval_query.query_interval_start_end_index(ts)
        if start_i is None:
            return None
        
        matched_tuple = self.data[start_i]

        # Check threshold if specified
        if self.coherent_time_sec is not None:
            closest_ts = matched_tuple[0]
            time_diff_sec = abs(TimestampUtil.diff(ts, closest_ts, unit='s'))
            if time_diff_sec > self.coherent_time_sec:
                return None
        
        # Use the value of the left closest record
        return matched_tuple[1]