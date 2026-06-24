from graph_workflow.Batch_Loader import batch
from models.schema import batch_schema


def main():

    Input = batch_schema(csv_path="Data/Inputs/tickets_2.csv", output_csv_path="")

    batch_schema.model_validate(batch.invoke(Input))


if __name__ == "__main__":
    main()
