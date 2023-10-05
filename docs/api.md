# API

## Accessors

```{eval-rst}
.. module:: ark.core._accessors
.. currentmodule:: ark.core

.. autosummary::
    :toctree: generated

    _accessors.accessors.register_spatial_data_accessor
    _accessors.accessors.SpatialDataAccessor
```

## Indexing and Coordinate Iteration (`.sel`, `.iter_coords`)

```{eval-rst}
.. module:: ark.core._indexing
.. currentmodule:: ark.core

.. autosummary::
    :toctree: generated

    _indexing.indexing.IndexingAccessor
    _indexing.indexing.IteratorAccessor
    _indexing.utils
```

## IO

```{eval-rst}
.. module:: ark.core._io
.. currentmodule:: ark.core

.. autosummary::
    :toctree: generated

    _io.io.load_cohort
    _io.io.convert_fov
    _io.io._fov
```

## Segmentation and Marker Quantification(`.segmentation`, `.marker_quantification`)

```{eval-rst}
.. module:: ark.core._segmentation
.. currentmodule:: ark.core

.. autosummary::
    :toctree: generated

    _segmentation.segmentation.SegmentationAccessor
    _segmentation.marker_quantification.MarkerQuantificationAccessor

    _segmentation.utils.deepcell.SegmentationImageContainer
    _segmentation.utils.deepcell._create_deepcell_input
    _segmentation.utils.deepcell.extract_zip
    _segmentation.utils.deepcell._deepcell_seg_to_spatial_labels
    _segmentation.utils.deepcell.spatial_data_to_fov
    _segmentation.utils.deepcell.zip_input_files
    _segmentation.utils.deepcell.upload_to_deepcell

    _segmentation.utils.regionprops_extraction.REGIONPROPS_BASE
    _segmentation.utils.regionprops_extraction.REGIONPROPS_BASE_TEMPORARY
    _segmentation.utils.regionprops_extraction.REGIONPROPS_SINGLE_COMP
    _segmentation.utils.regionprops_extraction.REGIONPROPS_MULTI_COMP
    _segmentation.utils.regionprops_extraction.DEFAULT_REGIONPROPS
    _segmentation.utils.regionprops_extraction.compute_region_props_df
    _segmentation.utils.regionprops_extraction.regionprops
    _segmentation.utils.regionprops_extraction._get_meta
    _segmentation.utils.regionprops_extraction._get_loop_sizes
    _segmentation.utils.regionprops_extraction._get_pos_core_dims
    _segmentation.utils.regionprops_extraction.rp_table_wrapper
    _segmentation.utils.regionprops_extraction.major_minor_axis_ratio
    _segmentation.utils.regionprops_extraction.perim_square_over_area
    _segmentation.utils.regionprops_extraction.major_axis_equiv_diam_ratio
    _segmentation.utils.regionprops_extraction.convex_hull_equiv_diam_ratio
    _segmentation.utils.regionprops_extraction.centroid_diff
    _segmentation.utils.regionprops_extraction.num_concavities
    _segmentation.utils.regionprops_extraction._diff_img_concavities
    _segmentation.utils.regionprops_extraction.nc_ratio

```
