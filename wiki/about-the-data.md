# Background on the Data

A little background on some of the terminology used throughout the data.

## Bills, Joint Resolutions, Simple Resolutions, etc.

What are all these different types of legislative proposals in Congress?

You'll notice there are a set of identifiers on the elements of the data set -- something like `hr` or `hjres` or `s`. These identifiers specify the precise type of legislative proposal.

The first character `h` or `s` indicates whether the proposal originated in the House or the Senate. Simple enough.

Then, it can be any one of the following types:

* **Bill**: Requires approval of both chambers of Congress and signature of the President. Become law.
* **Joint Resolution**: Effectively the same as a bill, except can be used to propose Constitutional amendment.
* **Concurrent Resolution**: Does not have the power of the law. Usually to change rules that affect both houses. Requires approval of both houses.
* **Simple Resolution**: Only requires approval of single house. Does not have power of the law.

The table below shows each identifier and what it refers to. 
| Identifier  |  Name       |
| ----------- | ----------- | 
| hr          | House Bill  |
| hjres       | House Joint Resolution |
| hconres | House Concurrent Resolution |
| hres | House Simple Resolution |
| s | Senate Bill |
|sjres | Senate Joint Resolution |
| sconres | Senate Concurrent Resolution |
| sres | Senate Simple Resolution |

## Additional Resources

* https://www.senate.gov/legislative/common/briefing/leg_laws_acts.htm
