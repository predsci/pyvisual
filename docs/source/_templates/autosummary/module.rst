{# expects pkgtree, fullname, name in context #}

{{ name | escape | underline}}

.. automodule:: {{ fullname }}
   :no-members:

.. currentmodule:: {{ fullname }}

{# start at the tree root #}
{% set ns = namespace(node=pkgtree, ok=True) %}
{% set match = namespace(value=none) %}
{% set route = fullname.split('.') %}

{# if the first segment is the root we're already on, skip it #}
{% if route and (route[0] == (ns.node['name'] if ns.node is mapping else ns.node.name)) %}
  {% set route = route[1:] %}
{% endif %}

{# walk down: find a child whose name matches seg AND is a package/module #}
{% for seg in route %}
  {% if ns.ok %}
    {% set match.value = none %}
    {% for c in (ns.node['children'] if ns.node is mapping else ns.node.children) %}
      {% set cname = c['name'] if c is mapping else c.name %}
      {% set ckind = c['kind'] if c is mapping else c.kind %}
      {% if cname == seg and (ckind == 'package' or ckind == 'module') %}
        {% set match.value = c %}
      {% endif %}
    {% endfor %}
    {% if match.value is not none %}
      {% set ns.node = match.value %}
    {% else %}
      {% set ns.ok = False %}
    {% endif %}
  {% endif %}
{% endfor %}

{% set here = ns.ok and ns.node %}

{# partition #}
{% set packages       = here.children | selectattr('kind','equalto','package')   | map(attribute='name') | sort | list %}
{% set modules        = here.children | selectattr('kind','equalto','module')    | map(attribute='name') | sort | list %}
{% set classes        = here.children | selectattr('kind','equalto','class')     | map(attribute='name') | sort | list %}
{% set functions      = here.children | selectattr('kind','equalto','function')  | map(attribute='name') | sort | list %}
{% set attributes     = here.children | selectattr('kind','equalto','attribute') | map(attribute='name') | sort | list %}
{% set exceptions     = here.children | selectattr('kind','equalto','exception') | map(attribute='name') | sort | list %}

{% if packages %}
.. rubric:: Subpackages

.. autosummary::
   :toctree: {{ name }}/
   :template: autosummary/package.rst
{% for item in packages %}
   {{ item }}
{%- endfor %}

{% endif %}

{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree: {{ name }}/
   :template: autosummary/module.rst
{% for item in modules %}
   {{ item }}
{%- endfor %}

{% endif %}

{% if classes %}
.. rubric:: Classes

.. autosummary::
   :toctree: {{ name }}/
   :template: autosummary/class.rst
{% for item in classes %}
   {{ item }}
{%- endfor %}

{% endif %}

{% if functions %}
.. rubric:: Functions

.. autosummary::
   :toctree: {{ name }}/
   :template: autosummary/function.rst
{% for item in functions %}
   {{ item }}
{%- endfor %}

{% endif %}

{% if attributes %}
.. rubric:: Attributes

.. autosummary::
   :toctree: {{ name }}/
   :template: autosummary/attribute.rst
{% for item in attributes %}
   {{ item }}
{%- endfor %}

{% endif %}

{% if exceptions %}
.. rubric:: Exceptions

.. autosummary::
   :toctree: {{ name }}/
   :template: autosummary/exception.rst
{% for item in exceptions %}
   {{ item }}
{%- endfor %}

{% endif %}
