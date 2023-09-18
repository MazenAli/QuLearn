from model_builder import QNNModel, ModelBuilder, QNNStatModel


def test_qnnmodel_creation():
    config = {
        "init_varlayer": True,
        "init_varlayer_nlayers": 1,
        "upload_type": "Sequential",
        "num_features": 4,
        "num_repeat_parallel": 1,
        "num_reuploads": 2,
        "num_varlayers": 2,
        "num_repeats": 3,
        "omega": 0.7,
        "hamiltonian_type": "AllWires",
        "id": 1,
    }
    model = QNNModel(**config)
    assert isinstance(model, QNNModel)
    assert model.num_features == config["num_features"]


def test_model_builder():
    builder = ModelBuilder()
    # Assume that model with id 1 exists in the JSON file
    config = builder.get_model_config(1)
    assert "num_features" in config
    assert "num_reuploads" in config
    model = builder.create_model(1)
    assert isinstance(model, QNNModel)


def test_model_builder_stat():
    builder = ModelBuilder(statistical=True)
    # Assume that model with id 1 exists in the JSON file
    config = builder.get_model_config(1)
    assert "num_features" in config
    assert "num_reuploads" in config
    model = builder.create_model(1)
    assert isinstance(model, QNNStatModel)
