# Todo

* Split C++ API Independent From Python API
* ~~Add head and tail function~~
* ~~Pretty Print Table~~
* Clean up the code
    * Add/remove `const` from places (I think I've been overly liberal here)
* Create util section
* Give C++ functions better names
* Indexing Function
    * DataFrame
        * Integer Row Indexing (iloc)
            * ~~Slice~~
            * ~~Int~~
        * Keyword Row Indexing (loc)
            * Slice
            * ~~Keyword~~
        * Column Indexing
            * `__getitem__`
                * ~~Keyword~~
            * ~~`__getattr__`~~
                * ~~Keyword~~
    * Series
        * Integer Row Indexing (iloc)
            * ~~Slice~~
            * ~~Int~~
        * Keyword Row Indexing (loc)
            * Slice
            * Keyword
* ~~Better options for combining slice with int~~
* More Indices
    * RangeIndex
    * DateTimeIndex
* DateTime object
* Fix the underscore names for a lot of these things 
    * Underscored things should be private/protected
* Label more things Private/Protected
* Fix the bug in how things are displayed when columns are even for a Series
* Add logging
* Add assertions (that can be turned off via macros)
* Allow other types other than double
* NDimFrame??? -> Difficult but doable
* ~~Add typedef for Eigen::Index~~
* Use index_t instead of Eigen::Index
* Use mask_t instead of slice<Eigen::Index>
* Create Parent Class (Frame) for DataFrame and Series