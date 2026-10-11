from src.validate_artifacts import validate


def test_artifacts_pass_validation():
    assert validate() == []
