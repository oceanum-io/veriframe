==========
Statistics
==========

Every statistic lives in :mod:`veriframe.stats` as a plain function of two
arrays, and is exposed on :class:`~veriframe.veriframe.VeriFrame` as a method
that supplies the reference and verification columns. The ones in the default
stats table are ``n``, ``bias``, ``rmsd``, ``si``, ``usi``, ``mad``, ``mrad``,
``nbias`` and ``nrmsd``.

.. note::

   ``bias``, ``rmsd``, ``mad`` and ``mrad`` accept a ``circular`` argument in
   :mod:`veriframe.stats`, but the corresponding **VeriFrame methods do not
   forward it** -- they are linear whatever you pass. For directions, call the
   functions in :mod:`veriframe.stats` directly with ``circular=True``.


Scatter: SI and USI
===================

Two indices describe the spread of the error, and they answer different
questions. Reporting only one of them regularly leads to the wrong conclusion.

Scatter Index (:func:`~veriframe.stats.si`)
-------------------------------------------

.. math::

   SI = \frac{\sqrt{\frac{1}{N}\sum_{i=1}^{N}
        \big[(B_i - \overline{B}) - (A_i - \overline{A})\big]^2}}{\overline{A}}

where :math:`A` is the observation and :math:`B` the model. This is the RMS
error with the **mean offset removed**, normalised by the observed mean. It is
the centred RMS difference of Taylor (2001).

Unexplained Scatter Index (:func:`~veriframe.stats.usi`)
---------------------------------------------------------

.. math::

   USI = \frac{\sigma_A \sqrt{1 - R^2}}{\overline{A}}

The RMS residual of the least-squares regression of the observations on the
model, normalised by the observed mean. It removes the mean offset **and** the
amplitude error.


Why both
--------

``SI`` removes the offset but not a gain error. Its square is the error
variance, which decomposes exactly (Murphy, 1988):

.. math::

   \operatorname{var}(B - A) = \underbrace{(\sigma_B - R\,\sigma_A)^2}_{\text{amplitude}}
                             + \underbrace{\sigma_A^2\,(1 - R^2)}_{\text{decorrelation}}

The first term is the amplitude error left once the model's correlation is
taken into account: the spread that would minimise the error is
:math:`R\,\sigma_A`, not :math:`\sigma_A`. Only the second term is scatter
in the everyday sense. A model that reproduces every event in phase but swings
25% too hard has no random error at all and still scores a large ``SI``.

``USI`` is exactly the decorrelation part, :math:`USI^2\,\overline{A}^2 =
\sigma_A^2\,(1 - R^2)`. It is invariant under any affine transform of the
model, :math:`B \rightarrow \alpha B + \beta`, so neither a bias nor a gain
error can move it. The test data shipped with the package, with the model
scaled up by 26%:

.. code-block:: python

   >>> vf.si(), vf.usi()
   (0.153, 0.114)          # SI inflated by the over-swing; USI unchanged
   >>> vf.mse_decomposition()["amplitude_share"]
   0.086                   # this part of the error variance is amplitude, not scatter

Read them together:

===================  ==================  =========================================
``SI``               ``USI``             interpretation
===================  ==================  =========================================
low                  low                 good
high                 low                 mis-scaled: gain error, not noise
high                 high                genuinely noisy, or timing errors
low                  high                not possible -- ``USI`` never exceeds
                                         ``SI``; they are equal only when
                                         :math:`\sigma_B = R\,\sigma_A`
===================  ==================  =========================================

``USI`` is built on the Pearson correlation and so applies to linear variables
only; there is deliberately no ``circular`` argument. It is NaN for a constant
model, where the correlation is undefined.


MSE decomposition
=================

:func:`~veriframe.stats.mse_decomposition` returns the three additive terms
directly:

.. math::

   MSE = \mathrm{bias}^2 + (\sigma_B - R\,\sigma_A)^2
         + \sigma_A^2\,(1 - R^2)

.. code-block:: python

   >>> d = vf.mse_decomposition()
   >>> {k: round(v, 3) for k, v in d.items() if k.endswith("share")}
   {'bias2_share': 0.113, 'amplitude_share': 0.001, 'decorrelation_share': 0.885}

Which term dominates is what says where to look: ``bias2`` at a mean offset,
``amplitude`` at a gain error such as a dissipation term scaling wrongly with
the magnitude of the signal, ``decorrelation`` at timing or forcing.

The identity is exact, so the terms sum to the MSE to within floating point,
and ``decorrelation`` equals :math:`(USI\,\overline{A})^2` by construction.


References
==========

Murphy, A.H. (1988). Skill scores based on the mean square error and their
relationships to the correlation coefficient. *Monthly Weather Review*,
116(12), 2417-2424.

Taylor, K.E. (2001). Summarizing multiple aspects of model performance in a
single diagram. *Journal of Geophysical Research*, 106(D7), 7183-7192.

Mentaschi, L., Besio, G., Cassola, F., Mazzino, A. (2013). Problems in
RMSE-based wave model validations. *Ocean Modelling*, 72, 53-58.

Hanna, S.R., Heinold, D.W. (1985). Development and application of a simple
method for evaluating air quality models. API Publication 4409, American
Petroleum Institute, Washington DC.
