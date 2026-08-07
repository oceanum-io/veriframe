=======
History
=======

Unreleased
----------

* Add ``stats.usi`` (Unexplained Scatter Index) and the matching
  ``VeriFrame.usi`` method. Where ``si`` removes only the mean offset, ``usi``
  also removes the amplitude error, so it is invariant under any affine
  transform of the model and isolates the genuinely unexplained scatter.
  ``USI`` is now a column of the default stats table.
* Add ``stats.mse_decomposition`` and ``VeriFrame.mse_decomposition``, the
  exact split of the mean square error into bias, amplitude and decorrelation
  terms that ``usi`` is derived from.

0.1.0 (2024-07-08)
------------------

* First release on PyPI.
