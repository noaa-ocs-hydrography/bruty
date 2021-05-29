import sys

import pytest

from nbs import configs

def test_plain():
    c = configs.load_config(r"configs\plain.config")
    assert c['DEFAULT']['load_test'] == 'testpassed'
    # c.write(sys.stdout)


def test_additional_configs():
    c = configs.load_config(r"configs\additional.config")
    assert c['DEFAULT']['load_test'] == 'testpassed_with_sub'
    # c.write(sys.stdout)


def test_custom_path():
    c = configs.load_config(r"custom_base_path.config", base_config_path=".\\configs")
    assert c['DEFAULT']['load_test'] == 'testpassed_local'
    # c.write(sys.stdout)


def test_custom_path_inside_config():
    c = configs.load_config(r"supply_base_path.config")
    assert c['DEFAULT']['load_test'] == 'testpassed_local'
    # c.write(sys.stdout)


def test_circular():
    with pytest.raises(FileExistsError):
        c = configs.load_config(r"configs\circular.config")
        assert c['DEFAULT']['load_test'] == 'testpassed'
        # c.write(sys.stdout)


def test_missing_files():
    with pytest.raises(FileNotFoundError):
        c = configs.load_config(r"configs\missing.config")
        assert c['DEFAULT']['load_test'] == 'testpassed'
        # c.write(sys.stdout)


def test_default_str():
    c = configs.load_config(r"configs\plain.config", initial_config=r"configs\default.config")
    assert c['DEFAULT']['some_text'] == 'test'
    # c.write(sys.stdout)


def test_default_parser():
    defcon = configs.configparser.ConfigParser()
    defcon.read(r"configs\default.config")
    c = configs.load_config(r"configs\plain.config", initial_config=defcon)
    assert c['DEFAULT']['load_test'] == 'testpassed'
    assert c['DEFAULT']['some_text'] == 'test'
    # c.write(sys.stdout)


def test_overwrite():
    c = configs.load_config(r"configs\overwrite.config")
    assert c['DEFAULT']['load_test'] == 'overwritepassed_with_sub'
    # c.write(sys.stdout)


def test_no_substitution():
    c = configs.load_config(r"configs\substitutions.config", interp=False, immediate_interp=False)
    # c.write(sys.stdout)
    assert c['DEFAULT']['subst'] == 'changed'
    assert c['DEFAULT']['sub_test'] == '${subst}${test_sub1}${test_sub2}'
    assert c['DEFAULT']['test_sub1'] == '${subst}'
    assert c['DEFAULT']['test_sub2'] == '${subst}'


def test_delay_substitution():
    c = configs.load_config(r"configs\substitutions.config", interp=True, immediate_interp=False)
    # c.write(sys.stdout)
    assert c['DEFAULT']['subst'] == 'changed'
    assert c['DEFAULT']['sub_test'] == 'changedchangedchanged'
    assert c['DEFAULT']['test_sub1'] == 'changed'
    assert c['DEFAULT']['test_sub2'] == 'changed'


def test_immediate_substitution():
    c = configs.load_config(r"configs\substitutions.config", interp=True, immediate_interp=True)
    # c.write(sys.stdout)
    assert c['DEFAULT']['subst'] == 'changed'
    assert c['DEFAULT']['sub_test'] == 'changedchangedbase'
    assert c['DEFAULT']['test_sub1'] == 'changed'
    assert c['DEFAULT']['test_sub2'] == 'base'

