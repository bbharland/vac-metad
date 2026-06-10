# Reloading Modules

```python
%load_ext autoreload
%autoreload 1
%aimport src.module_name
```

- `%autoreload 1` only reloads modules explicitly registered via
`%aimport`; everything else stays static.
- When switching focus to a different module, remind them to update the
registration, e.g. `%aimport -old_module` then `%aimport new_module`.
