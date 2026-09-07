# tests/test_ontology.py
from coremem.ontology import FactOntology


def test_unknown_attribute_defaults_to_multi_text():
    onto = FactOntology()
    a = onto.attribute("favorite_brand_of_soup")
    assert a.cardinality == "multi" and a.value_type == "text"


def test_starter_attributes_have_declared_cardinality():
    onto = FactOntology()
    assert onto.attribute("employer").cardinality == "single"
    assert onto.attribute("child").cardinality == "multi"


def test_toml_config_overrides_and_extends(tmp_path):
    cfg = tmp_path / "ontology.toml"
    cfg.write_text(
        '[attribute.airline]\ncardinality = "single"\nvalue_type = "text"\n'
        '[attribute.pet]\ncardinality = "multi"\n'
    )
    onto = FactOntology(config_path=str(cfg))
    assert onto.attribute("airline").cardinality == "single"
    assert onto.attribute("pet").cardinality == "multi"
    assert onto.attribute("employer").cardinality == "single"  # starter set survives merge