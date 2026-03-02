*===============================================================================
* 1. ENVIRONMENT SETUP
*===============================================================================
clear all
set more off
cd "."

*===============================================================================
* 2. DATA LOADING & RENAMING
*===============================================================================
* Load only required variables to conserve memory
use 股票代码 newipzlid 年份 标题 摘要 申请日 专利类型 IPC 被引证次数 ///
    using "data/patents.dta", clear

* Rename variables to English abbreviations for consistency
rename (股票代码 newipzlid 年份 标题 摘要 申请日 专利类型 IPC 被引证次数) ///
       (stkcd p_id p_year p_tt p_abs p_date p_type p_ipc p_cite)


*===============================================================================
* 3. DATA CLEANING
*===============================================================================
* Filter by patent type: keep invention applications, granted inventions, and utility models
keep if inlist(p_type, "发明申请", "发明授权", "实用新型")

* Filter by stock code prefix (0, 3, or 6)
* Handles both string and numeric formats without creating intermediate variables
capture confirm string variable stkcd
if _rc == 0 {
    * String format: directly extract first character
    keep if inlist(substr(stkcd, 1, 1), "0", "3", "6")
}
else {
    * Numeric format: pad to 6 digits with leading zeros, then extract first char
    keep if inlist(substr(string(stkcd, "%06.0f"), 1, 1), "0", "3", "6")
}


*===============================================================================
* 4. DATA VALIDATION
*===============================================================================
* Display sample size and patent type distribution
count
tabulate p_type, missing

* Check missing values
misstable summarize
count if missing(p_cite)

* Fill missing citations with 0
replace p_cite = 0 if missing(p_cite)

* Duplicate detection based on stock code and patent title
duplicates report stkcd p_id


*===============================================================================
* 5. DATA SAVING
*===============================================================================
* Save cleaned dataset to disk, overwriting existing file if present
save "data/patents_cleaned.dta", replace
