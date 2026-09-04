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
  exact split of the mean square error (Murphy, 1988) into bias, amplitude and
  decorrelation terms; the decorrelation term is what ``usi`` measures.
* Taylor diagram: scale the radial axis to the data instead of a fixed
  ``1.5 * refstd`` so no model is dropped off the edge, support negative
  correlations, warn about samples that cannot be drawn, draw on the figure
  and axes passed in rather than on pyplot's current ones, and use a
  colour-blind-safe default palette.
* Fix ``plot_set`` passing an unknown keyword to ``df2taylor`` so the model
  labels were silently ignored.

0.1.0 (2024-07-08)
------------------

* First release on PyPI.
