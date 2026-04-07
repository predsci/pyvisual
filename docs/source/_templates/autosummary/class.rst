{# ===== assumes pkgtree, fullname, name in context ===== #}

{{ name | escape | underline }}

.. autoclass:: {{ fullname }}
   :show-inheritance:
   :no-members:

{# ---------- walk to the containing package/module (like your working package code) ---------- #}
{% set ns = namespace(node=pkgtree, ok=True) %}
{% set match = namespace(value=none) %}
{% set route = fullname.split('.') %}

{# skip the root segment if it matches the tree root #}
{% if route and (route[0] == (ns.node['name'] if ns.node is mapping else ns.node.name)) %}
  {% set route = route[1:] %}
{% endif %}

{# split into container route + class name #}
{% set class_name = route[-1] %}
{% set pkg_route  = route[:-1] %}

{# descend to the container node (package/module) #}
{% for seg in pkg_route %}
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

{# locate the class/exception node among container's children #}
{% set container = ns.ok and ns.node %}
{% set clsnode = namespace(value=none) %}

{% if container %}
  {% for c in (container['children'] if container is mapping else container.children) %}
    {% set cname = c['name'] if c is mapping else c.name %}
    {% set ckind = c['kind'] if c is mapping else c.kind %}
    {% if cname == class_name and (ckind == 'class' or ckind == 'exception') %}
      {% set clsnode.value = c %}
    {% endif %}
  {% endfor %}
{% endif %}

{# ---------- partition class members (children) by kind ---------- #}
{% set kids = clsnode.value.children %}

{% set methods        = kids | selectattr('kind','equalto','method')        | map(attribute='name') | sort | list %}
{% set class_methods  = kids | selectattr('kind','equalto','class_method')  | map(attribute='name') | sort | list %}
{% set static_methods = kids | selectattr('kind','equalto','static_method') | map(attribute='name') | sort | list %}
{% set properties     = kids | selectattr('kind','equalto','property')      | map(attribute='name') | sort | list %}
{% set data           = kids | selectattr('kind','equalto','data')     | map(attribute='name') | sort | list %}
{% set descriptors    = kids | selectattr('kind','equalto','descriptor')    | map(attribute='name') | sort | list %}

{% if methods %}
.. rubric:: Methods

.. autosummary::
   :toctree: {{ name }}/
   :template: autosummary/method.rst
{% for item in methods %}
   ~{{ fullname ~ '.' ~ item }}
{%- endfor %}

{% endif %}

{% if class_methods %}
.. rubric:: Class Methods

.. autosummary::
   :toctree: {{ name }}/
   :template: autosummary/method.rst
{% for item in class_methods %}
   ~{{ fullname ~ '.' ~ item }}
{%- endfor %}

{% endif %}

{% if static_methods %}
.. rubric:: Static Methods

.. autosummary::
   :toctree: {{ name }}/
   :template: autosummary/method.rst
{% for item in static_methods %}
   ~{{ fullname ~ '.' ~ item }}
{%- endfor %}

{% endif %}

{% if properties %}
.. rubric:: Properties

.. autosummary::
   :toctree: {{ name }}/
   :template: autosummary/property.rst
{% for item in properties %}
   ~{{ fullname ~ '.' ~ item }}
{%- endfor %}

{% endif %}

{% if data %}
.. rubric:: Data

.. autosummary::
   :toctree: {{ name }}/
   :template: autosummary/data.rst
{% for item in data %}
   ~{{ fullname ~ '.' ~ item }}
{%- endfor %}

{% endif %}

{% if descriptors %}
.. rubric:: Descriptors

.. autosummary::
   :toctree: {{ name }}/
   :template: autosummary/attribute.rst
{% for item in descriptors %}
   ~{{ fullname ~ '.' ~ item }}
{%- endfor %}

{% endif %}
