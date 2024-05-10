.. _community_guide:

Community Guideline
===================

.. contents::
  :depth: 2
  :local:

Welcome to the MLC-LLM community! Just like you, all of us are in awe of the immense power of large language models.
Our goal for MLC-LLM is to foster a project that is driven by an open-source community, working together to democratize
this technology and make it accessible across various devices. We are thrilled to have you as part of our
community and eagerly anticipate your valuable contributions.


.. _community_discussion:

Participate in Community Discussions
------------------------------------

We encourage open discussions. If you encounter a bug or have a feature request, please file an issue in MLC-LLM's
GitHub `issue tracker <https://github.com/mlc-ai/mlc-llm/issues>`__. You are encouraged to tag the issue with labels
such as "bug," "feature request," or "iOS" so that the relevant developers can quickly notice your concern.

Additionally, we have set up a `discord server <https://discord.gg/9Xpy2HGBuD>`__ for online discussions.
While we encourage participation in the Discord server, we also recommend creating a GitHub issue even if the
topic has been discussed there. This ensures that the discussion is archived and searchable for future reference.

Before submitting an issue, we kindly ask you to check our :doc:`/community/faq` to see if your question has already been answered.

.. _contribute-to-mlc-llm:

Contribute to MLC-LLM
---------------------

.. _fork-and-create-pull-requests:

Fork and Create Pull Requests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ready to contribute to MLC-LLM? Awesome! We are excited to see you are ready to contribute your code.
The standard way to make changes to MLC-LLM code base is through creating a `pull-request <https://github.com/mlc-ai/mlc-llm/pulls>`__,
and we will review your code and merge it to the code base when it is ready.

The first step to becoming a developer is to `fork <https://github.com/mlc-ai/mlc-llm/fork>`__ the repository to your own
github account, you will notice a repository under ``https://github.com/username/mlc-llm`` where ``username`` is your github user name.

You can clone your fork to your local machine and commit changes, or edit the contents of your fork (in the case you are just fixing typos)
on GitHub directly. Once your update is complete, you can click the ``contribute`` button and open a pull request to the main repository.

.. _contribute-new-models:

Contribute New Models to MLC-LLM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* If you have compiled a model using our :doc:`/compilation/compile_models` tutorial for an existing model architecture, please upload your models to the internet (e.g., Hugging Face) by following :ref:`distribute-compiled-models` tutorial.

* If you add a new model variant to MLC-LLM by following our :doc:`/compilation/define_new_models` tutorial.
  Please create a pull request to add your model architecture (currently model architectures are placed under
  `relax_models <https://github.com/mlc-ai/mlc-llm/tree/main/mlc_llm/relax_model>`__ folder).

.. _coding-styles:

Coding Styles
^^^^^^^^^^^^^

For python codes, we generally follow the `PEP8 style guide <https://peps.python.org/pep-0008/>`__.
The python comments follow `NumPy style <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`__ python docstrings.
To make things easy, you can use `black <https://pypi.org/project/black/>`__ to automatically format your python code.

.. code:: bash

      pip install black
      black your_python_file.py

For C++ codes, we generally follow the `Google C++ style guide <https://google.github.io/styleguide/cppguide.html>`__.
The C++ comments should be `Doxygen compatible <http://www.doxygen.nl/manual/docblocks.html#cppblock>`__.
Fo your convenience, you can use `clang-format <https://clang.llvm.org/docs/ClangFormat.html>`__ to automatically format your C++ code.

.. code:: bash

      clang-format -i your_cpp_file.cpp

.. _general-development-process:

General Development Process
---------------------------

Everyone in the community is welcome to send patches, documents, and propose new directions to the project.
The key guideline here is to enable everyone in the community to get involved and participate in the decision and development.
We encourage public discussion in different channels, so that everyone in the community can participate
and get informed in developments.

Code reviews are one of the key ways to ensure the quality of the code. High-quality code reviews prevent technical debt
for long-term and are crucial to the success of the project. A pull request needs to be reviewed before it gets merged.
A committer who has the expertise of the corresponding area would moderate the pull request and merge the code when
it is ready. The corresponding committer could request multiple reviewers who are familiar with the area of the code.
We encourage contributors to request code reviews themselves and help review each other's code -- remember everyone
is volunteering their time to the community, high-quality code review itself costs as much as the actual code
contribution, you could get your code quickly reviewed if you do others the same favor.

The community should strive to reach a consensus on technical decisions through discussion. We expect committers to
moderate technical discussions in a diplomatic way, and provide suggestions with clear technical reasoning when necessary.


.. _roles-committers:

Committers
^^^^^^^^^^

Committers are individuals who are granted with write access to the project. A committer is usually responsible for
a certain area or several areas of the code where they oversee the code review process.
The area of contribution can take all forms, including code contributions and code reviews, documents, education, and outreach.
The review of pull requests will be assigned to the committers who recently contribute to the area this PR belongs to.
Committers are essential for a high quality and healthy project. The community actively looks for new committers
from contributors. Each existing committer can nominate new committers to MLC projects.

.. _roles-contributors:

Contributors
^^^^^^^^^^^^
We also welcome contributors if you are not ready to be a committer yet. Everyone who contributes to
the project (in the form of code, bugfix, documentation, tutorials, etc) is a contributor.
We maintain a `page <https://github.com/mlc-ai/mlc-llm/blob/main/CONTRIBUTORS.md>`__ to acknowledge contributors,
please let us know if you contribute to the project and if your name is not included in the list.
