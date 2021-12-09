from pathlib import Path
import great_expectations as ge
import pandas as pd

context = ge.data_context.DataContext()

#clinical-data
name="clinical-data"
context.create_expectation_suite(expectation_suite_name=name)
dataframe = pd.read_csv("../../../data/"+"csv/"+name+".csv")
df = ge.dataset.PandasDataset(dataframe)

df['dx1'] = df.dx1.replace('.',None)
# Unique
df.expect_column_values_to_be_unique(column="ADRC_ADRCCLINICALDATA ID")

# No null values
df.expect_column_values_to_not_be_null(column="ADRC_ADRCCLINICALDATA ID")
df.expect_column_values_to_not_be_null(column="Subject")
df.expect_column_values_to_not_be_null(column="ageAtEntry")
df.expect_column_values_to_not_be_null(column="dx1")

expectationSuite = df.get_expectation_suite()
#print(expectationSuite)
#print(df.validate(expectation_suite=expectationSuite, only_return_failures=True))
context.save_expectation_suite(expectation_suite=expectationSuite, expectation_suite_name=name)


#PUP
name="pup"
context.create_expectation_suite(expectation_suite_name=name)
dataframe = pd.read_csv("../../../data/"+"csv/"+name+".csv")
df = ge.dataset.PandasDataset(dataframe)
df.expect_column_values_to_not_be_null(column="PUP_PUPTIMECOURSEDATA ID")
df.expect_column_values_to_be_unique(column="PUP_PUPTIMECOURSEDATA ID")
df.expect_column_values_to_not_be_null(column="tracer")
df.expect_column_values_to_be_in_set(column="tracer",value_set=["AV45","PIB"])

expectationSuite = df.get_expectation_suite()
#print(expectationSuite)
#print(df.validate(expectation_suite=expectationSuite, only_return_failures=True))
context.save_expectation_suite(expectation_suite=expectationSuite, expectation_suite_name=name)


#labels
name="labels"
context.create_expectation_suite(expectation_suite_name=name)
dataframe = pd.read_csv("../../../data/"+name+".csv")
df = ge.dataset.PandasDataset(dataframe)
df.expect_column_values_to_not_be_null(column="PUP_PUPTIMECOURSEDATA ID")
df.expect_column_values_to_be_unique(column="PUP_PUPTIMECOURSEDATA ID")
df.expect_column_values_to_not_be_null(column="Date")
df.expect_column_values_to_not_be_null(column="tracer")
df.expect_column_values_to_not_be_null(column="Label")
df.expect_column_values_to_be_in_set(column="tracer",value_set=["AV45","PIB"])
df.expect_column_values_to_be_in_set(column="Label",value_set=[True,False])
df.expect_column_values_to_be_in_type_list(column="Label", type_list=["BOOLEAN","boolean","BOOL","TINYINT","BIT","bool","BooleanType"])

expectationSuite = df.get_expectation_suite()
#print(expectationSuite)
#print(df.validate(expectation_suite=expectationSuite, only_return_failures=True))
context.save_expectation_suite(expectation_suite=expectationSuite, expectation_suite_name=name)

#labelled_images
name="labelled-images"
context.create_expectation_suite(expectation_suite_name=name)
dataframe = pd.read_csv("../../../data/"+"prepared/"+name+".csv")
df = ge.dataset.PandasDataset(dataframe)

df.expect_column_values_to_not_be_null(column="filename")
df.expect_column_values_to_be_unique(column="filename")
df.expect_column_values_to_not_be_null(column="label")
df.expect_column_values_to_be_in_set(column="label",value_set=[True,False])
df.expect_column_values_to_be_in_type_list(column="label", type_list=["BOOLEAN","boolean","BOOL","TINYINT","BIT","bool","BooleanType"])

expectationSuite = df.get_expectation_suite()
#print(expectationSuite)
#print(df.validate(expectation_suite=expectationSuite, only_return_failures=True))
context.save_expectation_suite(expectation_suite=expectationSuite, expectation_suite_name=name)
