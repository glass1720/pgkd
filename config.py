from dataclasses import dataclass


@dataclass
class DatasetConfig:
    name: str
    hf_id: str  # HuggingFace hub id
    text_field: str
    label_field: str
    num_labels: int
    taxonomy: dict[int, str]  # {id: label_name}
    fewshot_k: int = 16


DATASETS: dict[str, DatasetConfig] = {
    "AG_NEWS": DatasetConfig(
        name="AG_NEWS",
        hf_id="ag_news",
        text_field="text",
        label_field="label",
        num_labels=4,
        taxonomy={0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
    ),
    "YahooAnswers": DatasetConfig(
        name="YahooAnswersTopics",
        hf_id="yahoo_answers_topics",
        text_field="text",
        label_field="label",
        num_labels=10,
        taxonomy={i: str(i) for i in range(10)},
    ),
    "HuffPost": DatasetConfig(
        name="HuffPost",
        hf_id="huffpost",
        text_field="headline",
        label_field="category",
        num_labels=41,
        taxonomy={...},
    ),
    "AMZN_Reviews": DatasetConfig(
        name="AmazonReviews",
        hf_id="amazon_polarity",
        text_field="content",
        label_field="label",
        num_labels=335,
        taxonomy={...},
    ),
}
