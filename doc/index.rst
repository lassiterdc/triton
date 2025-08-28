.. TRITON Documentation documentation main file

Welcome to TRITON
=================

The Two-dimensional Runoff Inundation Toolkit for Operational Needs (TRITON) is an open-source, computationally efficient 2D flood modeling toolkit that scales from a laptop to supercomputers. It solves the full shallow-water equations on CPUs and GPUs to produce rapid, reproducible inundation maps (see `Morales Hernández et al., 2021 <https://doi.org/10.1016/j.envsoft.2021.105034>`_ and :doc:`papers`).

Why TRITON
==========

Use TRITON for accurate and quick simulation of flood wave propagation and surface inundation.

.. grid:: 2
   :gutter: 2

   .. grid-item::
      :columns: 6

      - **Open-source, Physics-based hydrodynamic flood model**  
      - **Solves the full 2D shallow-water equations**   
      - **Accelerated on modern CPUs and GPUs, including multi-GPU systems**  
      - **Scales from a single laptop to leadership-class supercomputers**  
      - **Captures backwater effects and both fluvial (riverine) and pluvial (flash flood) events**     

   .. grid-item::
      :columns: 6

      .. only:: html

         .. raw:: html

            <video class="doc-anim" src="_static/trimmed.mp4"
                   autoplay loop muted playsinline controls
                   style="max-width:100%; height:auto;"></video>

      .. only:: latex

         .. image:: _static/triton_video_placeholder.png
            :alt: TRITON animation (see HTML version for video)

.. note::

   For details, see:

   Morales-Hernández, M., et al. (2021).  
   *TRITON: A Multi-GPU Open Source 2D Hydrodynamic Flood Model.*  
   *Environmental Modelling & Software, 141*, 105034.  
   https://doi.org/10.1016/j.envsoft.2021.105034



Quick Links
-----------

.. grid:: 5
   :gutter: 2

   .. grid-item-card:: Get Started
      :link: getting_started
      :link-type: doc
      :text-align: center

      Learn more about TRITON 

   .. grid-item-card:: Build & Install
      :link: cmake_arguments
      :link-type: doc
      :text-align: center

      Build from source or use containers.

   .. grid-item-card:: Case Studies
      :link: casestudy
      :link-type: doc
      :text-align: center

      Explore test cases and real world flood simulation.

   .. grid-item-card:: FAQ
      :link: faq
      :link-type: doc
      :text-align: center

      Answers to common questions.

   .. grid-item-card:: Support
      :link: https://triton.ornl.gov/contact/
      :text-align: center

      Get help or report an issue.



Showcase
========

TRITON supports a wide range of flood modeling applications. Below are a few representative examples.

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Large-scale flooding
      :text-align: center

      .. image:: _static/Missouri.png
         :alt: Large-scale flood extent
         :class: doc-hero

      Basin- and watershed-scale flood simulations on CPUs and GPUs.

   .. grid-item-card:: Forecasting
      :text-align: center

      .. image:: _static/triton_5.png
         :alt: Flood forecast animation
         :class: doc-hero

      Short-term flood forecasts for emergency response.

   .. grid-item-card:: Probabilistic mapping
      :text-align: center

      .. image:: _static/probabilistic.png
         :alt: Probabilistic flood hazard mapping
         :class: doc-hero

      Ensemble-based flood hazard and uncertainty maps.

   .. grid-item-card:: Dam-break scenarios
      :text-align: center

      .. image:: _static/taum_sauk_anim.gif
         :alt: Dam break flood wave
         :class: doc-hero

      High-resolution dam-break and levee breach simulations.


Relevant Papers & Studies
=========================

A curated list of publications related to TRITON and its applications is available at :doc:`papers`.

Project Website
===============

`TRITON <https://triton.ornl.gov>`_


.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   introduction
   before_you_run_TRITON
   

.. toctree::
   :maxdepth: 1
   :caption: Download and Build
   :hidden:

   installation
   cmake_arguments
   makefile
   docker
   machine_file

.. toctree::
   :maxdepth: 1
   :caption: Configure and Run
   :hidden:

   simulation_file
   casestudy
   tools

.. toctree::
   :maxdepth: 1
   :caption: Developer and API
   :hidden:

   devguide
   api
   docguide

.. toctree::
   :maxdepth: 1
   :caption: Resources
   :hidden:

   papers
