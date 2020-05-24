"""
Unit tests for ds_json class.
"""
import os
import numpy as np
import pandas as pd
import unittest
import bp
from pandas.util.testing import assert_frame_equal
from bp.ds_json import MyJson

class TestJson(unittest.TestCase):
    def test_parse_json_col(self):
        """Check one json example.
        """
        df = pd.DataFrame({'id': [0],
                'payload': ["""{"analytics": {"device": "Desktop",
                                                "email_open_rate_pct": 14.0},
                                "industry": "Construction",
                                "time_in_product_mins": 62.45}"""]
                })
        # required answer
        ans_req = pd.DataFrame({'id': [0],
            'industry': ['Construction'],
            'time_in_product_mins': [62.45],
            'analytics.device': ['Desktop'],
            'analytics.email_open_rate_pct': [14.0]})

        ans_req.index.name = 'index'

        # get answer from my modules
        mj = MyJson()
        ans = mj.parse_json_col(df,'payload')

        # assert equal
        assert_frame_equal(ans, ans_req)

        # check last value
        last_val = ans_req.iloc[0,-1]
        self.assertEqual(last_val,14.0)

if __name__ == '__main__':
    unittest.main()
