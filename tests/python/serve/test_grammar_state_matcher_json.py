# pylint: disable=missing-module-docstring,missing-function-docstring
# pylint: disable=redefined-outer-name,unbalanced-tuple-unpacking
"""This test uses the optimized JSON grammar provided by the grammar library."""
import sys
from typing import List, Optional

import pytest
import tvm
import tvm.testing
from tvm import TVMError

from mlc_chat.serve import BNFGrammar, GrammarStateMatcher
from mlc_chat.tokenizer import Tokenizer


@pytest.fixture(scope="function")
def json_grammar():
    return BNFGrammar.get_grammar_of_json()


(json_input_accepted,) = tvm.testing.parameters(
    ('{"name": "John"}',),
    ('{ "name" : "John" }',),
    ("{}",),
    ("[]",),
    ('{"name": "Alice", "age": 30, "city": "New York"}',),
    ('{"name": "Mike", "hobbies": ["reading", "cycling", "hiking"]}',),
    ('{"name": "Emma", "address": {"street": "Maple Street", "city": "Boston"}}',),
    ('[{"name": "David"}, {"name": "Sophia"}]',),
    (
        '{"name": "William", "age": null, "married": true, "children": ["Liam", "Olivia"],'
        ' "hasPets": false}',
    ),
    (
        '{"name": "Olivia", "contact": {"email": "olivia@example.com", "address": '
        '{"city": "Chicago", "zipcode": "60601"}}}',
    ),
    (
        '{"name": "Liam", "skills": ["Java", "Python"], "experience": '
        '[{"company": "CompanyA", "years": 5}, {"company": "CompanyB", "years": 3}]}',
    ),
    (
        '{"person": {"name": "Ethan", "age": 40}, "education": {"degree": "Masters", '
        '"university": "XYZ University"}, "work": [{"company": "ABC Corp", "position": '
        '"Manager"}, {"company": "DEF Corp", "position": "Senior Manager"}]}',
    ),
    (
        '{"name": "Charlotte", "details": {"personal": {"age": 35, "hobbies": ["gardening", '
        '"painting"]}, "professional": {"occupation": "Engineer", "skills": '
        '["CAD", "Project Management"], "projects": [{"name": "Project A", '
        '"status": "Completed"}, {"name": "Project B", "status": "In Progress"}]}}}',
    ),
)


def test_json_accept(json_grammar: BNFGrammar, json_input_accepted: str):
    assert GrammarStateMatcher(json_grammar).debug_match_complete_string(json_input_accepted)


(json_input_refused,) = tvm.testing.parameters(
    (r'{ name: "John" }',),
    (r'{ "name": "John" } ',),  # trailing space is not accepted
    (r'{ "name": "John", "age": 30, }',),
    (r'{ "name": "John", "address": { "street": "123 Main St", "city": "New York" }',),
    (r'{ "name": "John", "age": 30, "hobbies": ["reading", "traveling",], }',),
    (r'{ "name": "John", "age": 30.5.7 }',),
    (r'{ "name": "John, "age": 30, "hobbies": ["reading", "traveling"] }',),
    (
        r'{ "name": "John", "age": 30, "hobbies": ["reading", { "type": "outdoor", "list": '
        r'["hiking", "swimming",]}] }',
    ),
    (r'{ "name": "John", "age": 30, "status": "\P\J" }',),
    (
        r'{ "name": "John", "age": 30, "hobbies": ["reading", "traveling"], "address": '
        r'{ "street": "123 Main St", "city": "New York", "coordinates": { "latitude": 40.7128, '
        r'"longitude": -74.0060 }}}, "work": { "company": "Acme", "position": "developer" }}',
    ),
)


def test_json_refuse(json_grammar: BNFGrammar, json_input_refused):
    assert not GrammarStateMatcher(json_grammar).debug_match_complete_string(json_input_refused)


(json_input_pressure,) = tvm.testing.parameters(
    # Extra long string: 1k chars
    (
        '["Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer nec odio. Praesent '
        "libero. Sed cursus ante dapibus diam. Sed nisi. Nulla quis sem at nibh elementum "
        "imperdiet. Duis sagittis ipsum. Praesent mauris. Fusce nec tellus sed augue semper "
        "porta. Mauris massa. Vestibulum lacinia arcu eget nulla. Class aptent taciti sociosqu "
        "ad litora torquent per conubia nostra, per inceptos himenaeos. Curabitur sodales ligula "
        "in libero. Sed dignissim lacinia nunc. Curabitur tortor. Pellentesque nibh. Aenean quam. "
        "In scelerisque sem at dolor. Maecenas mattis. Sed convallis tristique sem. Proin ut "
        "ligula vel nunc egestas porttitor. Morbi lectus risus, iaculis vel, suscipit quis, "
        "luctus non, massa. Fusce ac turpis quis ligula lacinia aliquet. Mauris ipsum. Nulla "
        "metus metus, ullamcorper vel, tincidunt sed, euismod in, nibh. Quisque volutpat "
        "condimentum velit. Class aptent taciti sociosqu ad litora torquent per conubia nostra, "
        "per inceptos himenaeos. Nam nec ante. Sed lacinia, urna non tincidunt mattis, tortor "
        "neque adipiscing diam, a cursus ipsum ante quis turpis. Nulla facilisi. Ut fringilla. "
        "Suspendisse potenti. Nunc feugiat mi a tellus consequat imperdiet. Vestibulum sapien. "
        "Proin quam. Etiam ultrices. Suspendisse in justo eu magna luctus suscipit. Sed lectus. "
        "Integer euismod lacus luctus magna. Quisque cursus, metus vitae pharetra auctor, sem "
        'massa mattis sem, at interdum magna augue eget diam."]',
    ),
    # long and complex json: 3k chars
    (
        r"""{
    "web-app": {
    "servlet": [
        {
        "servlet-name": "cofaxCDS",
        "servlet-class": "org.cofax.cds.CDSServlet",
        "init-param": {
            "configGlossary:installationAt": "Philadelphia, PA",
            "configGlossary:adminEmail": "ksm@pobox.com",
            "configGlossary:poweredBy": "Cofax",
            "configGlossary:poweredByIcon": "/images/cofax.gif",
            "configGlossary:staticPath": "/content/static",
            "templateProcessorClass": "org.cofax.WysiwygTemplate",
            "templateLoaderClass": "org.cofax.FilesTemplateLoader",
            "templatePath": "templates",
            "templateOverridePath": "",
            "defaultListTemplate": "listTemplate.htm",
            "defaultFileTemplate": "articleTemplate.htm",
            "useJSP": false,
            "jspListTemplate": "listTemplate.jsp",
            "jspFileTemplate": "articleTemplate.jsp",
            "cachePackageTagsTrack": 200,
            "cachePackageTagsStore": 200,
            "cachePackageTagsRefresh": 60,
            "cacheTemplatesTrack": 100,
            "cacheTemplatesStore": 50,
            "cacheTemplatesRefresh": 15,
            "cachePagesTrack": 200,
            "cachePagesStore": 100,
            "cachePagesRefresh": 10,
            "cachePagesDirtyRead": 10,
            "searchEngineListTemplate": "forSearchEnginesList.htm",
            "searchEngineFileTemplate": "forSearchEngines.htm",
            "searchEngineRobotsDb": "WEB-INF/robots.db",
            "useDataStore": true,
            "dataStoreClass": "org.cofax.SqlDataStore",
            "redirectionClass": "org.cofax.SqlRedirection",
            "dataStoreName": "cofax",
            "dataStoreDriver": "com.microsoft.jdbc.sqlserver.SQLServerDriver",
            "dataStoreUrl": "jdbc:microsoft:sqlserver://LOCALHOST:1433;DatabaseName=goon",
            "dataStoreUser": "sa",
            "dataStorePassword": "dataStoreTestQuery",
            "dataStoreTestQuery": "SET NOCOUNT ON;select test='test';",
            "dataStoreLogFile": "/usr/local/tomcat/logs/datastore.log",
            "dataStoreInitConns": 10,
            "dataStoreMaxConns": 100,
            "dataStoreConnUsageLimit": 100,
            "dataStoreLogLevel": "debug",
            "maxUrlLength": 500
        }
        },
        {
        "servlet-name": "cofaxEmail",
        "servlet-class": "org.cofax.cds.EmailServlet",
        "init-param": {
            "mailHost": "mail1",
            "mailHostOverride": "mail2"
        }
        },
        {
        "servlet-name": "cofaxAdmin",
        "servlet-class": "org.cofax.cds.AdminServlet"
        },
        {
        "servlet-name": "fileServlet",
        "servlet-class": "org.cofax.cds.FileServlet"
        },
        {
        "servlet-name": "cofaxTools",
        "servlet-class": "org.cofax.cms.CofaxToolsServlet",
        "init-param": {
            "templatePath": "toolstemplates/",
            "log": 1,
            "logLocation": "/usr/local/tomcat/logs/CofaxTools.log",
            "logMaxSize": "",
            "dataLog": 1,
            "dataLogLocation": "/usr/local/tomcat/logs/dataLog.log",
            "dataLogMaxSize": "",
            "removePageCache": "/content/admin/remove?cache=pages&id=",
            "removeTemplateCache": "/content/admin/remove?cache=templates&id=",
            "fileTransferFolder": "/usr/local/tomcat/webapps/content/fileTransferFolder",
            "lookInContext": 1,
            "adminGroupID": 4,
            "betaServer": true
        }
        }
    ],
    "servlet-mapping": {
        "cofaxCDS": "/",
        "cofaxEmail": "/cofaxutil/aemail/*",
        "cofaxAdmin": "/admin/*",
        "fileServlet": "/static/*",
        "cofaxTools": "/tools/*"
    },
    "taglib": {
        "taglib-uri": "cofax.tld",
        "taglib-location": "/WEB-INF/tlds/cofax.tld"
    }
    }
}""",
    ),
)


def test_json_pressure(json_grammar: BNFGrammar, json_input_pressure):
    assert GrammarStateMatcher(json_grammar).debug_match_complete_string(json_input_pressure)


(input_find_rejected_tokens, expected_rejected_sizes) = tvm.testing.parameters(
    (
        # short test
        '{"id": 1,"name": "Example"}',
        [
            # fmt: off
            31989, 31912, 299, 299, 299, 31973, 31846, 31846, 31948, 31915, 299, 299, 299, 299,
            299, 31973, 31846, 31846, 292, 292, 292, 292, 292, 292, 292, 292, 31974, 31999
            # fmt: on
        ],
    ),
    (
        # long test
        """{
"id": 1,
"na": "ex",
"ac": true,
"t": ["t1", "t2"],
"ne": {"lv2": {"val": "dp"}, "arr": [1, 2, 3]},
"res": "res"
}""",
        [
            # fmt: off
            31989, 31912, 31912, 299, 299, 299, 31973, 31846, 31846, 31948, 31915, 31915, 299, 299,
            299, 31973, 31846, 31846, 292, 292, 292, 31974, 31915, 31915, 299, 299, 299, 31973,
            31846, 31846, 31997, 31997, 31998, 31974, 31915, 31915, 299, 299, 31973, 31846, 31846,
            31840, 291, 291, 291, 31969, 31846, 31846, 291, 291, 291, 31969, 31974, 31915, 31915,
            299, 299, 299, 31973, 31846, 31846, 31908, 299, 299, 299, 299, 31973, 31846, 31846,
            31906, 299, 299, 299, 299, 31973, 31846, 31846, 291, 291, 291, 31968, 31970, 31915,
            31915, 299, 299, 299, 299, 31973, 31846, 31846, 31840, 31943, 31846, 31846, 31943,
            31846, 31846, 31943, 31970, 31974, 31915, 31915, 299, 299, 299, 299, 31973, 31846,
            31846, 292, 292, 292, 292, 31974, 31974, 31999
            # fmt: on
        ],
    ),
)


def test_find_next_rejected_tokens(
    json_grammar: BNFGrammar,
    input_find_rejected_tokens: str,
    expected_rejected_sizes: Optional[List[int]] = None,
):
    tokenizer_path = "dist/Llama-2-7b-chat-hf-q4f16_1-MLC"
    tokenizer = Tokenizer(tokenizer_path)
    grammar_state_matcher = GrammarStateMatcher(json_grammar, tokenizer)

    real_sizes = []
    for c in input_find_rejected_tokens:
        rejected_token_ids = grammar_state_matcher.find_next_rejected_tokens()
        real_sizes.append(len(rejected_token_ids))
        print("Accepting char:", c, file=sys.stderr)
        assert grammar_state_matcher.debug_accept_char(ord(c))
    rejected_token_ids = grammar_state_matcher.find_next_rejected_tokens()
    real_sizes.append(len(rejected_token_ids))
    if expected_rejected_sizes is not None:
        assert real_sizes == expected_rejected_sizes


def test_token_based_operations(json_grammar: BNFGrammar):
    """Test accepting token and finding the next token mask."""
    token_table = [
        # fmt: off
        "<s>", "</s>", "a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", "\n", " ", '"a":true',
        # fmt: on
    ]
    input_splitted = ["{", '"', "abc", 'b"', ":", "6", ", ", " ", '"a":true', "}"]
    input_ids = [token_table.index(t) for t in input_splitted]

    grammar_state_matcher = GrammarStateMatcher(json_grammar, token_table)

    expected = [
        ["{"],
        ['"', "}", "\n", " ", '"a":true'],
        ["a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", " "],
        ["a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", " "],
        [":", "\n", " ", ':"'],
        ['"', "{", "6", "\n", " "],
        ["}", ", ", "6", "\n", " "],
        [" ", "\n", '"', '"a":true'],
        [" ", "\n", '"', '"a":true'],
        ["}", ", ", "\n", " "],
        ["</s>"],
    ]

    result = []

    for id in input_ids:
        rejected = grammar_state_matcher.find_next_rejected_tokens()
        accepted = list(set(range(len(token_table))) - set(rejected))
        accepted_tokens = [token_table[i] for i in accepted]
        result.append(accepted_tokens)
        assert id in accepted
        assert grammar_state_matcher.accept_token(id)

    rejected = grammar_state_matcher.find_next_rejected_tokens()
    accepted = list(set(range(len(token_table))) - set(rejected))
    accepted_tokens = [token_table[i] for i in accepted]
    result.append(accepted_tokens)

    assert result == expected


def test_rollback(json_grammar: BNFGrammar):
    token_table = [
        # fmt: off
        "<s>", "</s>", "a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", "\n", " ", '"a":true',
        # fmt: on
    ]
    input_splitted = ["{", '"', "abc", 'b"', ":", "6", ", ", " ", '"a":true', "}"]
    input_ids = [token_table.index(t) for t in input_splitted]

    grammar_state_matcher = GrammarStateMatcher(json_grammar, token_table, 5)

    assert grammar_state_matcher.max_rollback_steps() == 5

    input_ids_splitted = [input_ids[i : i + 2] for i in range(0, len(input_ids), 2)]

    for i_1, i_2 in input_ids_splitted:
        orig_result = []
        orig_result.append(grammar_state_matcher.find_next_rejected_tokens())
        assert grammar_state_matcher.accept_token(i_1)
        orig_result.append(grammar_state_matcher.find_next_rejected_tokens())
        assert grammar_state_matcher.accept_token(i_2)
        grammar_state_matcher.rollback(2)
        result_after_rollback = []
        result_after_rollback.append(grammar_state_matcher.find_next_rejected_tokens())
        assert grammar_state_matcher.accept_token(i_1)
        result_after_rollback.append(grammar_state_matcher.find_next_rejected_tokens())
        assert grammar_state_matcher.accept_token(i_2)
        assert orig_result == result_after_rollback


def test_reset(json_grammar: BNFGrammar):
    token_table = [
        # fmt: off
        "<s>", "</s>", "a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", "\n", " ", '"a":true',
        # fmt: on
    ]
    input_splitted = ["{", '"', "abc", 'b"', ":", "6", ", ", " ", '"a":true', "}"]
    input_ids = [token_table.index(t) for t in input_splitted]

    grammar_state_matcher = GrammarStateMatcher(json_grammar, token_table)

    orig_result = []

    for i in input_ids:
        orig_result.append(grammar_state_matcher.find_next_rejected_tokens())
        assert grammar_state_matcher.accept_token(i)

    grammar_state_matcher.reset_state()

    result_after_reset = []

    for i in input_ids:
        result_after_reset.append(grammar_state_matcher.find_next_rejected_tokens())
        assert grammar_state_matcher.accept_token(i)

    assert orig_result == result_after_reset


def test_termination(json_grammar: BNFGrammar):
    token_table = [
        # fmt: off
        "<s>", "</s>", "a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", "\n", " ", '"a":true',
        # fmt: on
    ]
    input_splitted = ["{", '"', "abc", 'b"', ":", "6", ", ", " ", '"a":true', "}", "</s>"]
    input_ids = [token_table.index(t) for t in input_splitted]

    grammar_state_matcher = GrammarStateMatcher(json_grammar, token_table, 5)

    orig_result = []

    for i in input_ids:
        orig_result.append(grammar_state_matcher.find_next_rejected_tokens())
        assert grammar_state_matcher.accept_token(i)

    assert grammar_state_matcher.is_terminated()

    with pytest.raises(TVMError):
        grammar_state_matcher.accept_token(0)

    with pytest.raises(TVMError):
        grammar_state_matcher.find_next_rejected_tokens()

    grammar_state_matcher.rollback(2)

    assert not grammar_state_matcher.is_terminated()
    assert grammar_state_matcher.accept_token(input_ids[-2])


if __name__ == "__main__":
    # Run a benchmark to show the performance before running tests
    test_find_next_rejected_tokens(BNFGrammar.get_grammar_of_json(), '{"id": 1,"name": "Example"}')

    tvm.testing.main()
