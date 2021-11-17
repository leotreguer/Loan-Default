# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
scored_data = dataiku.Dataset("scored_data")
scored_data_df = scored_data.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

test_output_dataset_df = scored_data_df # For this sample code, simply copy input to output

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
test_output_dataset_df["new_column"]=test_output_dataset_df["Amount_Requested"]/10

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
test_output_dataset = dataiku.Dataset("test_output_dataset")
test_output_dataset.write_with_schema(test_output_dataset_df)