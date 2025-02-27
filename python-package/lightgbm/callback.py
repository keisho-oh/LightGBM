# coding: utf-8
"""Callbacks library."""
import collections
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Union

from .basic import _ConfigAliases, _log_info, _log_warning

_EvalResultTuple = Union[
    List[Tuple[str, str, float, bool]],
    List[Tuple[str, str, float, bool, float]]
]


def _gt_delta(curr_score: float, best_score: float, delta: float) -> bool:
    return curr_score > best_score + delta


def _lt_delta(curr_score: float, best_score: float, delta: float) -> bool:
    return curr_score < best_score - delta


class EarlyStopException(Exception):
    """Exception of early stopping."""

    def __init__(self, best_iteration: int, best_score: _EvalResultTuple) -> None:
        """Create early stopping exception.

        Parameters
        ----------
        best_iteration : int
            The best iteration stopped.
        best_score : list of (eval_name, metric_name, eval_result, is_higher_better) tuple or (eval_name, metric_name, eval_result, is_higher_better, stdv) tuple
            Scores for each metric, on each validation set, as of the best iteration.
        """
        super().__init__()
        self.best_iteration = best_iteration
        self.best_score = best_score


# Callback environment used by callbacks
CallbackEnv = collections.namedtuple(
    "CallbackEnv",
    ["model",
     "params",
     "iteration",
     "begin_iteration",
     "end_iteration",
     "evaluation_result_list"])


def _format_eval_result(value: _EvalResultTuple, show_stdv: bool = True) -> str:
    """Format metric string."""
    if len(value) == 4:
        return f"{value[0]}'s {value[1]}: {value[2]:g}"
    elif len(value) == 5:
        if show_stdv:
            return f"{value[0]}'s {value[1]}: {value[2]:g} + {value[4]:g}"
        else:
            return f"{value[0]}'s {value[1]}: {value[2]:g}"
    else:
        raise ValueError("Wrong metric value")


def log_evaluation(period: int = 1, show_stdv: bool = True) -> Callable:
    """Create a callback that logs the evaluation results.

    By default, standard output resource is used.
    Use ``register_logger()`` function to register a custom logger.

    Note
    ----
    Requires at least one validation data.

    Parameters
    ----------
    period : int, optional (default=1)
        The period to log the evaluation results.
        The last boosting stage or the boosting stage found by using ``early_stopping`` callback is also logged.
    show_stdv : bool, optional (default=True)
        Whether to log stdv (if provided).

    Returns
    -------
    callback : callable
        The callback that logs the evaluation results every ``period`` boosting iteration(s).
    """
    def _callback(env: CallbackEnv) -> None:
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            result = '\t'.join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
            _log_info(f'[{env.iteration + 1}]\t{result}')
    _callback.order = 10  # type: ignore
    return _callback


def record_evaluation(eval_result: Dict[str, Dict[str, List[Any]]]) -> Callable:
    """Create a callback that records the evaluation history into ``eval_result``.

    Parameters
    ----------
    eval_result : dict
        Dictionary used to store all evaluation results of all validation sets.
        This should be initialized outside of your call to ``record_evaluation()`` and should be empty.
        Any initial contents of the dictionary will be deleted.

        .. rubric:: Example

        With two validation sets named 'eval' and 'train', and one evaluation metric named 'logloss'
        this dictionary after finishing a model training process will have the following structure:

        .. code-block::

            {
             'train':
                 {
                  'logloss': [0.48253, 0.35953, ...]
                 },
             'eval':
                 {
                  'logloss': [0.480385, 0.357756, ...]
                 }
            }

    Returns
    -------
    callback : callable
        The callback that records the evaluation history into the passed dictionary.
    """
    if not isinstance(eval_result, dict):
        raise TypeError('eval_result should be a dictionary')
    eval_result.clear()

    def _init(env: CallbackEnv) -> None:
        for data_name, eval_name, _, _ in env.evaluation_result_list:
            eval_result.setdefault(data_name, collections.OrderedDict())
            eval_result[data_name].setdefault(eval_name, [])

    def _callback(env: CallbackEnv) -> None:
        if not eval_result:
            _init(env)
        for data_name, eval_name, result, _ in env.evaluation_result_list:
            eval_result[data_name][eval_name].append(result)
    _callback.order = 20  # type: ignore
    return _callback


def reset_parameter(**kwargs: Union[list, Callable]) -> Callable:
    """Create a callback that resets the parameter after the first iteration.

    .. note::

        The initial parameter will still take in-effect on first iteration.

    Parameters
    ----------
    **kwargs : value should be list or callable
        List of parameters for each boosting round
        or a callable that calculates the parameter in terms of
        current number of round (e.g. yields learning rate decay).
        If list lst, parameter = lst[current_round].
        If callable func, parameter = func(current_round).

    Returns
    -------
    callback : callable
        The callback that resets the parameter after the first iteration.
    """
    def _callback(env: CallbackEnv) -> None:
        new_parameters = {}
        for key, value in kwargs.items():
            if isinstance(value, list):
                if len(value) != env.end_iteration - env.begin_iteration:
                    raise ValueError(f"Length of list {key!r} has to equal to 'num_boost_round'.")
                new_param = value[env.iteration - env.begin_iteration]
            else:
                new_param = value(env.iteration - env.begin_iteration)
            if new_param != env.params.get(key, None):
                new_parameters[key] = new_param
        if new_parameters:
            env.model.reset_parameter(new_parameters)
            env.params.update(new_parameters)
    _callback.before_iteration = True  # type: ignore
    _callback.order = 10  # type: ignore
    return _callback


def early_stopping(stopping_rounds: int, first_metric_only: bool = False, verbose: bool = True, min_delta: Union[float, List[float]] = 0.0) -> Callable:
    """Create a callback that activates early stopping.

    Activates early stopping.
    The model will train until the validation score doesn't improve by at least ``min_delta``.
    Validation score needs to improve at least every ``stopping_rounds`` round(s)
    to continue training.
    Requires at least one validation data and one metric.
    If there's more than one, will check all of them. But the training data is ignored anyway.
    To check only the first metric set ``first_metric_only`` to True.
    The index of iteration that has the best performance will be saved in the ``best_iteration`` attribute of a model.

    Parameters
    ----------
    stopping_rounds : int
        The possible number of rounds without the trend occurrence.
    first_metric_only : bool, optional (default=False)
        Whether to use only the first metric for early stopping.
    verbose : bool, optional (default=True)
        Whether to log message with early stopping information.
        By default, standard output resource is used.
        Use ``register_logger()`` function to register a custom logger.
    min_delta : float or list of float, optional (default=0.0)
        Minimum improvement in score to keep training.
        If float, this single value is used for all metrics.
        If list, its length should match the total number of metrics.

    Returns
    -------
    callback : callable
        The callback that activates early stopping.
    """
    best_score = []
    best_iter = []
    best_score_list: list = []
    cmp_op = []
    enabled = True
    inited = False
    first_metric = ''

    def _init(env: CallbackEnv) -> None:
        nonlocal best_score
        nonlocal best_iter
        nonlocal best_score_list
        nonlocal cmp_op
        nonlocal enabled
        nonlocal inited
        nonlocal first_metric
        enabled = not any(env.params.get(boost_alias, "") == 'dart' for boost_alias
                          in _ConfigAliases.get("boosting"))
        if not enabled:
            _log_warning('Early stopping is not available in dart mode')
            return
        if not env.evaluation_result_list:
            raise ValueError('For early stopping, '
                             'at least one dataset and eval metric is required for evaluation')

        if verbose:
            _log_info(f"Training until validation scores don't improve for {stopping_rounds} rounds")

        # reset storages
        best_score = []
        best_iter = []
        best_score_list = []
        cmp_op = []
        inited = True
        first_metric = ''

        n_metrics = len(set(m[1] for m in env.evaluation_result_list))
        n_datasets = len(env.evaluation_result_list) // n_metrics
        if isinstance(min_delta, list):
            if not all(t >= 0 for t in min_delta):
                raise ValueError('Values for early stopping min_delta must be non-negative.')
            if len(min_delta) == 0:
                if verbose:
                    _log_info('Disabling min_delta for early stopping.')
                deltas = [0.0] * n_datasets * n_metrics
            elif len(min_delta) == 1:
                if verbose:
                    _log_info(f'Using {min_delta[0]} as min_delta for all metrics.')
                deltas = min_delta * n_datasets * n_metrics
            else:
                if len(min_delta) != n_metrics:
                    raise ValueError('Must provide a single value for min_delta or as many as metrics.')
                if first_metric_only and verbose:
                    _log_info(f'Using only {min_delta[0]} as early stopping min_delta.')
                deltas = min_delta * n_datasets
        else:
            if min_delta < 0:
                raise ValueError('Early stopping min_delta must be non-negative.')
            if min_delta > 0 and n_metrics > 1 and not first_metric_only and verbose:
                _log_info(f'Using {min_delta} as min_delta for all metrics.')
            deltas = [min_delta] * n_datasets * n_metrics

        # split is needed for "<dataset type> <metric>" case (e.g. "train l1")
        first_metric = env.evaluation_result_list[0][1].split(" ")[-1]
        for eval_ret, delta in zip(env.evaluation_result_list, deltas):
            best_iter.append(0)
            best_score_list.append(None)
            if eval_ret[3]:  # greater is better
                best_score.append(float('-inf'))
                cmp_op.append(partial(_gt_delta, delta=delta))
            else:
                best_score.append(float('inf'))
                cmp_op.append(partial(_lt_delta, delta=delta))

    def _final_iteration_check(env: CallbackEnv, eval_name_splitted: List[str], i: int) -> None:
        nonlocal best_iter
        nonlocal best_score_list
        nonlocal inited
        if env.iteration == env.end_iteration - 1:
            if verbose:
                best_score_str = '\t'.join([_format_eval_result(x) for x in best_score_list[i]])
                _log_info('Did not meet early stopping. '
                          f'Best iteration is:\n[{best_iter[i] + 1}]\t{best_score_str}')
                if first_metric_only:
                    _log_info(f"Evaluated only: {eval_name_splitted[-1]}")
            inited = False
            raise EarlyStopException(best_iter[i], best_score_list[i])

    def _callback(env: CallbackEnv) -> None:
        nonlocal best_score
        nonlocal best_iter
        nonlocal best_score_list
        nonlocal cmp_op
        nonlocal enabled
        nonlocal inited
        nonlocal first_metric
        if not inited:
            _init(env)
        if not enabled:
            return
        for i in range(len(env.evaluation_result_list)):
            score = env.evaluation_result_list[i][2]
            if best_score_list[i] is None or cmp_op[i](score, best_score[i]):
                best_score[i] = score
                best_iter[i] = env.iteration
                best_score_list[i] = env.evaluation_result_list
            # split is needed for "<dataset type> <metric>" case (e.g. "train l1")
            eval_name_splitted = env.evaluation_result_list[i][1].split(" ")
            if first_metric_only and first_metric != eval_name_splitted[-1]:
                continue  # use only the first metric for early stopping
            if ((env.evaluation_result_list[i][0] == "cv_agg" and eval_name_splitted[0] == "train"
                 or env.evaluation_result_list[i][0] == env.model._train_data_name)):
                _final_iteration_check(env, eval_name_splitted, i)
                continue  # train data for lgb.cv or sklearn wrapper (underlying lgb.train)
            elif env.iteration - best_iter[i] >= stopping_rounds:
                if verbose:
                    eval_result_str = '\t'.join([_format_eval_result(x) for x in best_score_list[i]])
                    _log_info(f"Early stopping, best iteration is:\n[{best_iter[i] + 1}]\t{eval_result_str}")
                    if first_metric_only:
                        _log_info(f"Evaluated only: {eval_name_splitted[-1]}")
                inited = False
                raise EarlyStopException(best_iter[i], best_score_list[i])
            _final_iteration_check(env, eval_name_splitted, i)
    _callback.order = 30  # type: ignore
    return _callback
