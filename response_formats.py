from pydantic import BaseModel


class GeneratedSample(BaseModel):
    text: str
    label: int
    label_name: str


class GeneratedSamples(BaseModel):
    samples: list[GeneratedSample]


class ClassDecsripiton(BaseModel):
    label: int
    label_name: str
    description: str


class Taxonomy(BaseModel):
    classes: list[ClassDecsripiton]
