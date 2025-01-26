from pydantic import BaseModel, Field


class ModelBenchmarks(BaseModel):
    """
    Performance benchmarks for comparing different models.
    """

    mmlu_score: float
    gsm8k_score: float | None = None
    bbh_score: float | None = None


class ModelLatency(BaseModel):
    """
    Latency benchmarks for comparing different models.
    """

    time_to_first_token_ms: float = Field(gt=0)
    """ 
    Median Time to first token in milliseconds.
    """

    tokens_per_second: float = Field(gt=0)
    """
    Median output tokens per second.
    """


class ModelCost(BaseModel):
    """
    Cost benchmarks for comparing different models.
    """

    input_cost_per_1m: float = Field(gt=0)
    """
    Cost per 1M input tokens.
    """

    output_cost_per_1m: float = Field(gt=0)
    """
    Cost per 1M output tokens.
    """


class ModelMetrics(BaseModel):
    """
    Model metrics for comparing different models.
    """

    cost: ModelCost
    speed: ModelLatency
    intelligence: ModelBenchmarks


class ModelInfo(BaseModel):
    name: str
    description: str | None = None
    provider: str
    metrics: ModelMetrics
