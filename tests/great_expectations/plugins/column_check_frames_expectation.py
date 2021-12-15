from great_expectations.core import ExpectationConfiguration, ExpectationValidationResult
from great_expectations.execution_engine import (
   ExecutionEngine,
   PandasExecutionEngine,
   SparkDFExecutionEngine,
   SqlAlchemyExecutionEngine,
)
from great_expectations.expectations.expectation import ColumnExpectation
from great_expectations.expectations.metrics import (
   ColumnMetricProvider,
   column_aggregate_value, column_aggregate_partial,
   column_map_metrics,
)
from great_expectations.expectations.metrics.import_manager import F, sa
from great_expectations.expectations.util import render_evaluation_parameter_string
from great_expectations.render.renderer.renderer import renderer
from great_expectations.render.types import RenderedStringTemplateContent, RenderedTableContent, RenderedBulletListContent, RenderedGraphContent
from great_expectations.render.util import substitute_none_for_missing 

from typing import Any, Dict, List, Optional, Union

import os      
from pathlib import Path

class ExpectColumnToBeValidPath(ColumnExpectation):
    # Setting necessary computation metric dependencies and defining kwargs, as well as assigning kwargs default values
    metric_dependencies = ("column_values.in_set")

    def _validate(
        self,
        configuration: ExpectationConfiguration,
        metrics: Dict,
        runtime_configuration: dict = None,
        execution_engine: ExecutionEngine = None,
    ):

        in_set = metrics["column_values.in_set"]

        base_path = Path(__file__).parent
        dir_path = (base_path / "../../../../data/raw/").resolve()

        value_set = os.listdir(dir_path)

        return {"success": in_set, "result": {"observed_value": in_set}}
