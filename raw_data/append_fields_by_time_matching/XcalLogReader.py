
import glob
import os
import pandas as pd


class XcalLogReader:
    FIELD_TIMESTAMP = 'TIME_STAMP'
    FIELD_UTC_TS = 'utc_ts'

    # TODO: add a function to read xcal logs and convert the timestamp to UTC timestamp
    # TODO: add a function to read xcal logs and convert the timestamp to UTC timestamp
    """
    Read xcal logs and convert the timestamp to UTC timestamp
    :base_dir: the directory of the xcal logs
    :parsing_timezone: the timezone at which xcal logs are parsed, e.g. 'US/Eastern'
    """
    def __init__(self, base_dir: str, parsing_timezone: str):
        self.base_dir = base_dir
        self._parsing_timezone = parsing_timezone

    def get_parsing_timezone(self):
        return self._parsing_timezone
    
    def set_parsing_timezone(self, parsing_timezone: str):
        self._parsing_timezone = parsing_timezone

    def get_xcal_logs_of_operator(self, operator: str):
        files = glob.glob(os.path.join(self.base_dir, f'{operator}*.xlsx'))
        files = sorted(files)
        return files
    
    def convert_xcal_timestamp_to_utc_ts(self, df: pd.DataFrame):
        dt_series = pd.to_datetime(
            df[self.FIELD_TIMESTAMP], 
            errors='coerce',
        )
        # These changes will make the timezone conversion more robust by:
        # - Handling ambiguous times during the fall DST transition (when clocks are set back)
        # - Handling non-existent times during the spring DST transition (when clocks are set forward)
        # - Preventing errors from being raised during these transitions
        df[self.FIELD_UTC_TS] = dt_series.dt.tz_localize(
            self._parsing_timezone,
            ambiguous='infer',
            nonexistent='shift_forward'
        ).dt.tz_convert('UTC')

        df = df.dropna(subset=[self.FIELD_UTC_TS])
        df[self.FIELD_UTC_TS] = df[self.FIELD_UTC_TS].apply(lambda x: int(x.timestamp() * 1000))
        df = df.sort_values(by=self.FIELD_UTC_TS, ascending=True)
        return df

    def read_xcal_log(self, xcal_file: str):
        if xcal_file.endswith('.xlsx'):
            df = pd.read_excel(xcal_file)
        elif xcal_file.endswith('.csv'):
            df = pd.read_csv(xcal_file)
        else:
            raise ValueError(f'Unsupported file extension: {xcal_file}, only .xlsx and .csv are supported')
        df = self.convert_xcal_timestamp_to_utc_ts(df)
        return df