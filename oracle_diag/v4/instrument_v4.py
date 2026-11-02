"""Capture the first ten real loss values without changing gradients or RNG."""


def install_loss_trace(model):
    trace = []

    def record(name, value):
        if model.total_it <= 10:
            scalar = float(value.detach().cpu())
            trace.append(dict(step=int(model.total_it), loss=name, value=scalar))

    if hasattr(model, 'combine_target_source_loss'):
        original = model.combine_target_source_loss
        calls = [0]

        def combined(*args, **kwargs):
            value = original(*args, **kwargs)
            name = ('V', 'Q', 'actor')[calls[0] % 3]
            calls[0] += 1
            record(name, value)
            return value
        model.combine_target_source_loss = combined
    else:
        for attr, name in [('update_v_function','V'), ('update_q_functions','Q'), ('update_policy','actor')]:
            original = getattr(model, attr)

            def wrapped(*args, _original=original, _name=name, **kwargs):
                value = _original(*args, **kwargs)
                record(_name, value[0] if isinstance(value, tuple) else value)
                return value
            setattr(model, attr, wrapped)
    return trace
