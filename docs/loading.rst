FCS dataset loading
===================

We provide a classe to load and manage large flow cytometry datasets:

:FCDataSet: Reperesents a set of FCS observations. 
Behaves like a list and stores only the path of the FCS files providing an efficient way to manage large datasets.
The instanciation of the FCMeasurement object corresponding to a FCS file is done by indexing the FCDataSet instance.
Used with the scikit-learn-like :doc:`SampleWisePipeline` from the bactoml library it also provide an efficient ways to 
construct pipeline to process whole datasets.

See :doc:`api` for usage.