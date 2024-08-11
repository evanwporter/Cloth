# Todo

* Split C++ API Independent From Python API
* ~~Pretty Print Table~~
* Clean up the code
    * Add/remove `const` from places (I think I've been overly liberal here)
* Create util section
* Give C++ functions better names
* Add expanded # indexing functions
    * ~~Column index that returns series for DF (operator [])~~
        * ~~Converts to `__getattr___` & `__getitem___` in python~~
    * Loc Proxy for both
* Better options for combing slice with int
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
* NDimFrame??? -> Difficult but doable
