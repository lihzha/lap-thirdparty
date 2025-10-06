import dataclasses

from openpi import transforms as upstream_transforms

from openpi_cot.models.adapters.model_adapter import ExtendedModelType
from openpi_cot.policies.cot_policy import CoTInputs


# TODO: during inference, inputs need to be converted to the same encoding as the model first, normalize, and then convert to robot-acceptable encoding.
@dataclasses.dataclass(frozen=True)
class VQAInputs(CoTInputs):
    # Determines which model will be used.
    action_dim: int = 32
    model_type: ExtendedModelType = ExtendedModelType.PI_COT

    def __call__(self, data: dict) -> dict:
        return super()._prepare_inputs(data)


@dataclasses.dataclass(frozen=True)
class VQAOutputs(upstream_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        return {"reasoning": data.get("reasoning")}
