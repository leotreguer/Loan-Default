# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
scored_data = dataiku.Dataset("scored_data")
scored_data_df = scored_data.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
scored_data_df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

scored_data_processed_df = scored_data_df # For this sample code, simply copy input to output

#test


# Write recipe outputs
scored_data_processed = dataiku.Dataset("scored_data_processed")
scored_data_processed.write_with_schema(scored_data_processed_df)