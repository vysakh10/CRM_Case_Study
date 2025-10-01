from pydantic import BaseModel, Field


class FileArgs(BaseModel):
    customers_path: str = Field(
        ..., description="The filepath which contains customer details"
    )

    non_customers_path: str = Field(
        ..., description="The filepath which contains non-customer details"
    )

    usage_path: str = Field(
        ..., description="The filepath which contains usage data of users."
    )
