??
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.22v2.8.2-0-g2ea19cbb5758??
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
conv2d_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_21/kernel
}
$conv2d_21/kernel/Read/ReadVariableOpReadVariableOpconv2d_21/kernel*&
_output_shapes
:*
dtype0
t
conv2d_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_21/bias
m
"conv2d_21/bias/Read/ReadVariableOpReadVariableOpconv2d_21/bias*
_output_shapes
:*
dtype0
?
conv2d_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_22/kernel
}
$conv2d_22/kernel/Read/ReadVariableOpReadVariableOpconv2d_22/kernel*&
_output_shapes
:*
dtype0
t
conv2d_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_22/bias
m
"conv2d_22/bias/Read/ReadVariableOpReadVariableOpconv2d_22/bias*
_output_shapes
:*
dtype0
?
conv2d_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*!
shared_nameconv2d_23/kernel
}
$conv2d_23/kernel/Read/ReadVariableOpReadVariableOpconv2d_23/kernel*&
_output_shapes
:(*
dtype0
t
conv2d_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_nameconv2d_23/bias
m
"conv2d_23/bias/Read/ReadVariableOpReadVariableOpconv2d_23/bias*
_output_shapes
:(*
dtype0
?
conv2d_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:((*!
shared_nameconv2d_24/kernel
}
$conv2d_24/kernel/Read/ReadVariableOpReadVariableOpconv2d_24/kernel*&
_output_shapes
:((*
dtype0
t
conv2d_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_nameconv2d_24/bias
m
"conv2d_24/bias/Read/ReadVariableOpReadVariableOpconv2d_24/bias*
_output_shapes
:(*
dtype0
y
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?-<*
shared_namedense_6/kernel
r
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes
:	?-<*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:<*
dtype0
y
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	<?-*
shared_namedense_7/kernel
r
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes
:	<?-*
dtype0
q
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?-*
shared_namedense_7/bias
j
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes	
:?-*
dtype0
?
conv2d_transpose_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:((**
shared_nameconv2d_transpose_6/kernel
?
-conv2d_transpose_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_6/kernel*&
_output_shapes
:((*
dtype0
?
conv2d_transpose_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*(
shared_nameconv2d_transpose_6/bias

+conv2d_transpose_6/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_6/bias*
_output_shapes
:(*
dtype0
?
conv2d_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:((*!
shared_nameconv2d_25/kernel
}
$conv2d_25/kernel/Read/ReadVariableOpReadVariableOpconv2d_25/kernel*&
_output_shapes
:((*
dtype0
t
conv2d_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_nameconv2d_25/bias
m
"conv2d_25/bias/Read/ReadVariableOpReadVariableOpconv2d_25/bias*
_output_shapes
:(*
dtype0
?
conv2d_transpose_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:(**
shared_nameconv2d_transpose_7/kernel
?
-conv2d_transpose_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_7/kernel*&
_output_shapes
:(*
dtype0
?
conv2d_transpose_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_7/bias

+conv2d_transpose_7/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_7/bias*
_output_shapes
:*
dtype0
?
conv2d_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_26/kernel
}
$conv2d_26/kernel/Read/ReadVariableOpReadVariableOpconv2d_26/kernel*&
_output_shapes
:*
dtype0
t
conv2d_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_26/bias
m
"conv2d_26/bias/Read/ReadVariableOpReadVariableOpconv2d_26/bias*
_output_shapes
:*
dtype0
?
conv2d_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_27/kernel
}
$conv2d_27/kernel/Read/ReadVariableOpReadVariableOpconv2d_27/kernel*&
_output_shapes
:*
dtype0
t
conv2d_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_27/bias
m
"conv2d_27/bias/Read/ReadVariableOpReadVariableOpconv2d_27/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/conv2d_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_21/kernel/m
?
+Adam/conv2d_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_21/bias/m
{
)Adam/conv2d_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_22/kernel/m
?
+Adam/conv2d_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_22/bias/m
{
)Adam/conv2d_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*(
shared_nameAdam/conv2d_23/kernel/m
?
+Adam/conv2d_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/kernel/m*&
_output_shapes
:(*
dtype0
?
Adam/conv2d_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*&
shared_nameAdam/conv2d_23/bias/m
{
)Adam/conv2d_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/bias/m*
_output_shapes
:(*
dtype0
?
Adam/conv2d_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:((*(
shared_nameAdam/conv2d_24/kernel/m
?
+Adam/conv2d_24/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/kernel/m*&
_output_shapes
:((*
dtype0
?
Adam/conv2d_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*&
shared_nameAdam/conv2d_24/bias/m
{
)Adam/conv2d_24/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/bias/m*
_output_shapes
:(*
dtype0
?
Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?-<*&
shared_nameAdam/dense_6/kernel/m
?
)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m*
_output_shapes
:	?-<*
dtype0
~
Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*$
shared_nameAdam/dense_6/bias/m
w
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
_output_shapes
:<*
dtype0
?
Adam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	<?-*&
shared_nameAdam/dense_7/kernel/m
?
)Adam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/m*
_output_shapes
:	<?-*
dtype0

Adam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?-*$
shared_nameAdam/dense_7/bias/m
x
'Adam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/m*
_output_shapes	
:?-*
dtype0
?
 Adam/conv2d_transpose_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:((*1
shared_name" Adam/conv2d_transpose_6/kernel/m
?
4Adam/conv2d_transpose_6/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_6/kernel/m*&
_output_shapes
:((*
dtype0
?
Adam/conv2d_transpose_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*/
shared_name Adam/conv2d_transpose_6/bias/m
?
2Adam/conv2d_transpose_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_6/bias/m*
_output_shapes
:(*
dtype0
?
Adam/conv2d_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:((*(
shared_nameAdam/conv2d_25/kernel/m
?
+Adam/conv2d_25/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_25/kernel/m*&
_output_shapes
:((*
dtype0
?
Adam/conv2d_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*&
shared_nameAdam/conv2d_25/bias/m
{
)Adam/conv2d_25/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_25/bias/m*
_output_shapes
:(*
dtype0
?
 Adam/conv2d_transpose_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*1
shared_name" Adam/conv2d_transpose_7/kernel/m
?
4Adam/conv2d_transpose_7/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_7/kernel/m*&
_output_shapes
:(*
dtype0
?
Adam/conv2d_transpose_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_7/bias/m
?
2Adam/conv2d_transpose_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_7/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_26/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_26/kernel/m
?
+Adam/conv2d_26/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_26/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_26/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_26/bias/m
{
)Adam/conv2d_26/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_26/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_27/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_27/kernel/m
?
+Adam/conv2d_27/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_27/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_27/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_27/bias/m
{
)Adam/conv2d_27/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_27/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_21/kernel/v
?
+Adam/conv2d_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_21/bias/v
{
)Adam/conv2d_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_22/kernel/v
?
+Adam/conv2d_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_22/bias/v
{
)Adam/conv2d_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*(
shared_nameAdam/conv2d_23/kernel/v
?
+Adam/conv2d_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/kernel/v*&
_output_shapes
:(*
dtype0
?
Adam/conv2d_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*&
shared_nameAdam/conv2d_23/bias/v
{
)Adam/conv2d_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/bias/v*
_output_shapes
:(*
dtype0
?
Adam/conv2d_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:((*(
shared_nameAdam/conv2d_24/kernel/v
?
+Adam/conv2d_24/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/kernel/v*&
_output_shapes
:((*
dtype0
?
Adam/conv2d_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*&
shared_nameAdam/conv2d_24/bias/v
{
)Adam/conv2d_24/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/bias/v*
_output_shapes
:(*
dtype0
?
Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?-<*&
shared_nameAdam/dense_6/kernel/v
?
)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v*
_output_shapes
:	?-<*
dtype0
~
Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*$
shared_nameAdam/dense_6/bias/v
w
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
_output_shapes
:<*
dtype0
?
Adam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	<?-*&
shared_nameAdam/dense_7/kernel/v
?
)Adam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/v*
_output_shapes
:	<?-*
dtype0

Adam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?-*$
shared_nameAdam/dense_7/bias/v
x
'Adam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/v*
_output_shapes	
:?-*
dtype0
?
 Adam/conv2d_transpose_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:((*1
shared_name" Adam/conv2d_transpose_6/kernel/v
?
4Adam/conv2d_transpose_6/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_6/kernel/v*&
_output_shapes
:((*
dtype0
?
Adam/conv2d_transpose_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*/
shared_name Adam/conv2d_transpose_6/bias/v
?
2Adam/conv2d_transpose_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_6/bias/v*
_output_shapes
:(*
dtype0
?
Adam/conv2d_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:((*(
shared_nameAdam/conv2d_25/kernel/v
?
+Adam/conv2d_25/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_25/kernel/v*&
_output_shapes
:((*
dtype0
?
Adam/conv2d_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*&
shared_nameAdam/conv2d_25/bias/v
{
)Adam/conv2d_25/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_25/bias/v*
_output_shapes
:(*
dtype0
?
 Adam/conv2d_transpose_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*1
shared_name" Adam/conv2d_transpose_7/kernel/v
?
4Adam/conv2d_transpose_7/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_7/kernel/v*&
_output_shapes
:(*
dtype0
?
Adam/conv2d_transpose_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_7/bias/v
?
2Adam/conv2d_transpose_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_7/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_26/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_26/kernel/v
?
+Adam/conv2d_26/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_26/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_26/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_26/bias/v
{
)Adam/conv2d_26/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_26/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_27/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_27/kernel/v
?
+Adam/conv2d_27/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_27/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_27/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_27/bias/v
{
)Adam/conv2d_27/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_27/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
Ŕ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
layer_with_weights-4
layer-8
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
 layer_with_weights-3
 layer-5
!layer_with_weights-4
!layer-6
"layer_with_weights-5
"layer-7
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses*
?
)iter

*beta_1

+beta_2
	,decay
-learning_rate.m?/m?0m?1m?2m?3m?4m?5m?6m?7m?8m?9m?:m?;m?<m?=m?>m??m?@m?Am?Bm?Cm?.v?/v?0v?1v?2v?3v?4v?5v?6v?7v?8v?9v?:v?;v?<v?=v?>v??v?@v?Av?Bv?Cv?*
?
.0
/1
02
13
24
35
46
57
68
79
810
911
:12
;13
<14
=15
>16
?17
@18
A19
B20
C21*
?
.0
/1
02
13
24
35
46
57
68
79
810
911
:12
;13
<14
=15
>16
?17
@18
A19
B20
C21*
* 
?
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
* 
* 
* 

Iserving_default* 
?

.kernel
/bias
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses*
?

0kernel
1bias
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses*
?
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z_random_generator
[__call__
*\&call_and_return_all_conditional_losses* 
?

2kernel
3bias
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses*
?

4kernel
5bias
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses*
?
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m_random_generator
n__call__
*o&call_and_return_all_conditional_losses* 
?
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses* 
?

6kernel
7bias
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses*
J
.0
/1
02
13
24
35
46
57
68
79*
J
.0
/1
02
13
24
35
46
57
68
79*
* 
?
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
?

8kernel
9bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?

:kernel
;bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?

<kernel
=bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?

>kernel
?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?

@kernel
Abias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?

Bkernel
Cbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
Z
80
91
:2
;3
<4
=5
>6
?7
@8
A9
B10
C11*
Z
80
91
:2
;3
<4
=5
>6
?7
@8
A9
B10
C11*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_21/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_21/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_22/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_22/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_23/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_23/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_24/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_24/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_6/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_6/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_7/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_7/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2d_transpose_6/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose_6/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEconv2d_25/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_25/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2d_transpose_7/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose_7/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEconv2d_26/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_26/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEconv2d_27/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_27/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

?0*
* 
* 
* 

.0
/1*

.0
/1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses*
* 
* 

00
11*

00
11*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses* 
* 
* 
* 

20
31*

20
31*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses*
* 
* 

40
51*

40
51*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
i	variables
jtrainable_variables
kregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses* 
* 
* 

60
71*

60
71*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses*
* 
* 
* 
C
0
1
2
3
4
5
6
7
8*
* 
* 
* 

80
91*

80
91*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 

:0
;1*

:0
;1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

<0
=1*

<0
=1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

>0
?1*

>0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

@0
A1*

@0
A1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

B0
C1*

B0
C1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
<
0
1
2
3
4
 5
!6
"7*
* 
* 
* 
<

?total

?count
?	variables
?	keras_api*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
sm
VARIABLE_VALUEAdam/conv2d_21/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_21/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_22/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_22/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_23/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_23/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_24/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_24/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_6/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/dense_6/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_7/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_7/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_6/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_6/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv2d_25/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_25/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_7/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_7/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv2d_26/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_26/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv2d_27/kernel/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_27/bias/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_21/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_21/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_22/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_22/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_23/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_23/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_24/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_24/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_6/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/dense_6/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_7/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_7/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_6/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_6/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv2d_25/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_25/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_7/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_7/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv2d_26/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_26/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv2d_27/kernel/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_27/bias/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
serving_default_encoder_inputPlaceholder*/
_output_shapes
:?????????00*
dtype0*$
shape:?????????00
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_encoder_inputconv2d_21/kernelconv2d_21/biasconv2d_22/kernelconv2d_22/biasconv2d_23/kernelconv2d_23/biasconv2d_24/kernelconv2d_24/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasconv2d_transpose_6/kernelconv2d_transpose_6/biasconv2d_25/kernelconv2d_25/biasconv2d_transpose_7/kernelconv2d_transpose_7/biasconv2d_26/kernelconv2d_26/biasconv2d_27/kernelconv2d_27/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_382060
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp$conv2d_21/kernel/Read/ReadVariableOp"conv2d_21/bias/Read/ReadVariableOp$conv2d_22/kernel/Read/ReadVariableOp"conv2d_22/bias/Read/ReadVariableOp$conv2d_23/kernel/Read/ReadVariableOp"conv2d_23/bias/Read/ReadVariableOp$conv2d_24/kernel/Read/ReadVariableOp"conv2d_24/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOp-conv2d_transpose_6/kernel/Read/ReadVariableOp+conv2d_transpose_6/bias/Read/ReadVariableOp$conv2d_25/kernel/Read/ReadVariableOp"conv2d_25/bias/Read/ReadVariableOp-conv2d_transpose_7/kernel/Read/ReadVariableOp+conv2d_transpose_7/bias/Read/ReadVariableOp$conv2d_26/kernel/Read/ReadVariableOp"conv2d_26/bias/Read/ReadVariableOp$conv2d_27/kernel/Read/ReadVariableOp"conv2d_27/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/conv2d_21/kernel/m/Read/ReadVariableOp)Adam/conv2d_21/bias/m/Read/ReadVariableOp+Adam/conv2d_22/kernel/m/Read/ReadVariableOp)Adam/conv2d_22/bias/m/Read/ReadVariableOp+Adam/conv2d_23/kernel/m/Read/ReadVariableOp)Adam/conv2d_23/bias/m/Read/ReadVariableOp+Adam/conv2d_24/kernel/m/Read/ReadVariableOp)Adam/conv2d_24/bias/m/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOp)Adam/dense_7/kernel/m/Read/ReadVariableOp'Adam/dense_7/bias/m/Read/ReadVariableOp4Adam/conv2d_transpose_6/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_6/bias/m/Read/ReadVariableOp+Adam/conv2d_25/kernel/m/Read/ReadVariableOp)Adam/conv2d_25/bias/m/Read/ReadVariableOp4Adam/conv2d_transpose_7/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_7/bias/m/Read/ReadVariableOp+Adam/conv2d_26/kernel/m/Read/ReadVariableOp)Adam/conv2d_26/bias/m/Read/ReadVariableOp+Adam/conv2d_27/kernel/m/Read/ReadVariableOp)Adam/conv2d_27/bias/m/Read/ReadVariableOp+Adam/conv2d_21/kernel/v/Read/ReadVariableOp)Adam/conv2d_21/bias/v/Read/ReadVariableOp+Adam/conv2d_22/kernel/v/Read/ReadVariableOp)Adam/conv2d_22/bias/v/Read/ReadVariableOp+Adam/conv2d_23/kernel/v/Read/ReadVariableOp)Adam/conv2d_23/bias/v/Read/ReadVariableOp+Adam/conv2d_24/kernel/v/Read/ReadVariableOp)Adam/conv2d_24/bias/v/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOp)Adam/dense_7/kernel/v/Read/ReadVariableOp'Adam/dense_7/bias/v/Read/ReadVariableOp4Adam/conv2d_transpose_6/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_6/bias/v/Read/ReadVariableOp+Adam/conv2d_25/kernel/v/Read/ReadVariableOp)Adam/conv2d_25/bias/v/Read/ReadVariableOp4Adam/conv2d_transpose_7/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_7/bias/v/Read/ReadVariableOp+Adam/conv2d_26/kernel/v/Read/ReadVariableOp)Adam/conv2d_26/bias/v/Read/ReadVariableOp+Adam/conv2d_27/kernel/v/Read/ReadVariableOp)Adam/conv2d_27/bias/v/Read/ReadVariableOpConst*V
TinO
M2K	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_383018
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv2d_21/kernelconv2d_21/biasconv2d_22/kernelconv2d_22/biasconv2d_23/kernelconv2d_23/biasconv2d_24/kernelconv2d_24/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasconv2d_transpose_6/kernelconv2d_transpose_6/biasconv2d_25/kernelconv2d_25/biasconv2d_transpose_7/kernelconv2d_transpose_7/biasconv2d_26/kernelconv2d_26/biasconv2d_27/kernelconv2d_27/biastotalcountAdam/conv2d_21/kernel/mAdam/conv2d_21/bias/mAdam/conv2d_22/kernel/mAdam/conv2d_22/bias/mAdam/conv2d_23/kernel/mAdam/conv2d_23/bias/mAdam/conv2d_24/kernel/mAdam/conv2d_24/bias/mAdam/dense_6/kernel/mAdam/dense_6/bias/mAdam/dense_7/kernel/mAdam/dense_7/bias/m Adam/conv2d_transpose_6/kernel/mAdam/conv2d_transpose_6/bias/mAdam/conv2d_25/kernel/mAdam/conv2d_25/bias/m Adam/conv2d_transpose_7/kernel/mAdam/conv2d_transpose_7/bias/mAdam/conv2d_26/kernel/mAdam/conv2d_26/bias/mAdam/conv2d_27/kernel/mAdam/conv2d_27/bias/mAdam/conv2d_21/kernel/vAdam/conv2d_21/bias/vAdam/conv2d_22/kernel/vAdam/conv2d_22/bias/vAdam/conv2d_23/kernel/vAdam/conv2d_23/bias/vAdam/conv2d_24/kernel/vAdam/conv2d_24/bias/vAdam/dense_6/kernel/vAdam/dense_6/bias/vAdam/dense_7/kernel/vAdam/dense_7/bias/v Adam/conv2d_transpose_6/kernel/vAdam/conv2d_transpose_6/bias/vAdam/conv2d_25/kernel/vAdam/conv2d_25/bias/v Adam/conv2d_transpose_7/kernel/vAdam/conv2d_transpose_7/bias/vAdam/conv2d_26/kernel/vAdam/conv2d_26/bias/vAdam/conv2d_27/kernel/vAdam/conv2d_27/bias/v*U
TinN
L2J*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_383247??
?%
?
C__inference_model_7_layer_call_and_return_conditional_losses_380990

inputs!
dense_7_380907:	<?-
dense_7_380909:	?-3
conv2d_transpose_6_380928:(('
conv2d_transpose_6_380930:(*
conv2d_25_380945:((
conv2d_25_380947:(3
conv2d_transpose_7_380950:('
conv2d_transpose_7_380952:*
conv2d_26_380967:
conv2d_26_380969:*
conv2d_27_380984:
conv2d_27_380986:
identity??!conv2d_25/StatefulPartitionedCall?!conv2d_26/StatefulPartitionedCall?!conv2d_27/StatefulPartitionedCall?*conv2d_transpose_6/StatefulPartitionedCall?*conv2d_transpose_7/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCallinputsdense_7_380907dense_7_380909*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_380906?
reshape_3/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_reshape_3_layer_call_and_return_conditional_losses_380926?
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall"reshape_3/PartitionedCall:output:0conv2d_transpose_6_380928conv2d_transpose_6_380930*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_380837?
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0conv2d_25_380945conv2d_25_380947*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_25_layer_call_and_return_conditional_losses_380944?
*conv2d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0conv2d_transpose_7_380950conv2d_transpose_7_380952*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_380882?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_7/StatefulPartitionedCall:output:0conv2d_26_380967conv2d_26_380969*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_26_layer_call_and_return_conditional_losses_380966?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0conv2d_27_380984conv2d_27_380986*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_27_layer_call_and_return_conditional_losses_380983?
IdentityIdentity*conv2d_27/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall+^conv2d_transpose_6/StatefulPartitionedCall+^conv2d_transpose_7/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????<: : : : : : : : : : : : 2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2X
*conv2d_transpose_7/StatefulPartitionedCall*conv2d_transpose_7/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?
?
3__inference_conv2d_transpose_7_layer_call_fn_382702

inputs!
unknown:(
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_380882?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????(: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????(
 
_user_specified_nameinputs
??
?
G__inference_autoencoder_layer_call_and_return_conditional_losses_382009

inputsJ
0model_6_conv2d_21_conv2d_readvariableop_resource:?
1model_6_conv2d_21_biasadd_readvariableop_resource:J
0model_6_conv2d_22_conv2d_readvariableop_resource:?
1model_6_conv2d_22_biasadd_readvariableop_resource:J
0model_6_conv2d_23_conv2d_readvariableop_resource:(?
1model_6_conv2d_23_biasadd_readvariableop_resource:(J
0model_6_conv2d_24_conv2d_readvariableop_resource:((?
1model_6_conv2d_24_biasadd_readvariableop_resource:(A
.model_6_dense_6_matmul_readvariableop_resource:	?-<=
/model_6_dense_6_biasadd_readvariableop_resource:<A
.model_7_dense_7_matmul_readvariableop_resource:	<?->
/model_7_dense_7_biasadd_readvariableop_resource:	?-]
Cmodel_7_conv2d_transpose_6_conv2d_transpose_readvariableop_resource:((H
:model_7_conv2d_transpose_6_biasadd_readvariableop_resource:(J
0model_7_conv2d_25_conv2d_readvariableop_resource:((?
1model_7_conv2d_25_biasadd_readvariableop_resource:(]
Cmodel_7_conv2d_transpose_7_conv2d_transpose_readvariableop_resource:(H
:model_7_conv2d_transpose_7_biasadd_readvariableop_resource:J
0model_7_conv2d_26_conv2d_readvariableop_resource:?
1model_7_conv2d_26_biasadd_readvariableop_resource:J
0model_7_conv2d_27_conv2d_readvariableop_resource:?
1model_7_conv2d_27_biasadd_readvariableop_resource:
identity??(model_6/conv2d_21/BiasAdd/ReadVariableOp?'model_6/conv2d_21/Conv2D/ReadVariableOp?(model_6/conv2d_22/BiasAdd/ReadVariableOp?'model_6/conv2d_22/Conv2D/ReadVariableOp?(model_6/conv2d_23/BiasAdd/ReadVariableOp?'model_6/conv2d_23/Conv2D/ReadVariableOp?(model_6/conv2d_24/BiasAdd/ReadVariableOp?'model_6/conv2d_24/Conv2D/ReadVariableOp?&model_6/dense_6/BiasAdd/ReadVariableOp?%model_6/dense_6/MatMul/ReadVariableOp?(model_7/conv2d_25/BiasAdd/ReadVariableOp?'model_7/conv2d_25/Conv2D/ReadVariableOp?(model_7/conv2d_26/BiasAdd/ReadVariableOp?'model_7/conv2d_26/Conv2D/ReadVariableOp?(model_7/conv2d_27/BiasAdd/ReadVariableOp?'model_7/conv2d_27/Conv2D/ReadVariableOp?1model_7/conv2d_transpose_6/BiasAdd/ReadVariableOp?:model_7/conv2d_transpose_6/conv2d_transpose/ReadVariableOp?1model_7/conv2d_transpose_7/BiasAdd/ReadVariableOp?:model_7/conv2d_transpose_7/conv2d_transpose/ReadVariableOp?&model_7/dense_7/BiasAdd/ReadVariableOp?%model_7/dense_7/MatMul/ReadVariableOp?
'model_6/conv2d_21/Conv2D/ReadVariableOpReadVariableOp0model_6_conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
model_6/conv2d_21/Conv2DConv2Dinputs/model_6/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
(model_6/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp1model_6_conv2d_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_6/conv2d_21/BiasAddBiasAdd!model_6/conv2d_21/Conv2D:output:00model_6/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00|
model_6/conv2d_21/TanhTanh"model_6/conv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00?
'model_6/conv2d_22/Conv2D/ReadVariableOpReadVariableOp0model_6_conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
model_6/conv2d_22/Conv2DConv2Dmodel_6/conv2d_21/Tanh:y:0/model_6/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
(model_6/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp1model_6_conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_6/conv2d_22/BiasAddBiasAdd!model_6/conv2d_22/Conv2D:output:00model_6/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????|
model_6/conv2d_22/TanhTanh"model_6/conv2d_22/BiasAdd:output:0*
T0*/
_output_shapes
:?????????d
model_6/dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
model_6/dropout_6/dropout/MulMulmodel_6/conv2d_22/Tanh:y:0(model_6/dropout_6/dropout/Const:output:0*
T0*/
_output_shapes
:?????????i
model_6/dropout_6/dropout/ShapeShapemodel_6/conv2d_22/Tanh:y:0*
T0*
_output_shapes
:?
6model_6/dropout_6/dropout/random_uniform/RandomUniformRandomUniform(model_6/dropout_6/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype0m
(model_6/dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
&model_6/dropout_6/dropout/GreaterEqualGreaterEqual?model_6/dropout_6/dropout/random_uniform/RandomUniform:output:01model_6/dropout_6/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:??????????
model_6/dropout_6/dropout/CastCast*model_6/dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:??????????
model_6/dropout_6/dropout/Mul_1Mul!model_6/dropout_6/dropout/Mul:z:0"model_6/dropout_6/dropout/Cast:y:0*
T0*/
_output_shapes
:??????????
'model_6/conv2d_23/Conv2D/ReadVariableOpReadVariableOp0model_6_conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype0?
model_6/conv2d_23/Conv2DConv2D#model_6/dropout_6/dropout/Mul_1:z:0/model_6/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
paddingSAME*
strides
?
(model_6/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp1model_6_conv2d_23_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
model_6/conv2d_23/BiasAddBiasAdd!model_6/conv2d_23/Conv2D:output:00model_6/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(|
model_6/conv2d_23/TanhTanh"model_6/conv2d_23/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(?
'model_6/conv2d_24/Conv2D/ReadVariableOpReadVariableOp0model_6_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
:((*
dtype0?
model_6/conv2d_24/Conv2DConv2Dmodel_6/conv2d_23/Tanh:y:0/model_6/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
paddingSAME*
strides
?
(model_6/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp1model_6_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
model_6/conv2d_24/BiasAddBiasAdd!model_6/conv2d_24/Conv2D:output:00model_6/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(|
model_6/conv2d_24/TanhTanh"model_6/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(d
model_6/dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
model_6/dropout_7/dropout/MulMulmodel_6/conv2d_24/Tanh:y:0(model_6/dropout_7/dropout/Const:output:0*
T0*/
_output_shapes
:?????????(i
model_6/dropout_7/dropout/ShapeShapemodel_6/conv2d_24/Tanh:y:0*
T0*
_output_shapes
:?
6model_6/dropout_7/dropout/random_uniform/RandomUniformRandomUniform(model_6/dropout_7/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????(*
dtype0m
(model_6/dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
&model_6/dropout_7/dropout/GreaterEqualGreaterEqual?model_6/dropout_7/dropout/random_uniform/RandomUniform:output:01model_6/dropout_7/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????(?
model_6/dropout_7/dropout/CastCast*model_6/dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????(?
model_6/dropout_7/dropout/Mul_1Mul!model_6/dropout_7/dropout/Mul:z:0"model_6/dropout_7/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????(h
model_6/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
model_6/flatten_3/ReshapeReshape#model_6/dropout_7/dropout/Mul_1:z:0 model_6/flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????-?
%model_6/dense_6/MatMul/ReadVariableOpReadVariableOp.model_6_dense_6_matmul_readvariableop_resource*
_output_shapes
:	?-<*
dtype0?
model_6/dense_6/MatMulMatMul"model_6/flatten_3/Reshape:output:0-model_6/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<?
&model_6/dense_6/BiasAdd/ReadVariableOpReadVariableOp/model_6_dense_6_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0?
model_6/dense_6/BiasAddBiasAdd model_6/dense_6/MatMul:product:0.model_6/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<?
%model_7/dense_7/MatMul/ReadVariableOpReadVariableOp.model_7_dense_7_matmul_readvariableop_resource*
_output_shapes
:	<?-*
dtype0?
model_7/dense_7/MatMulMatMul model_6/dense_6/BiasAdd:output:0-model_7/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????-?
&model_7/dense_7/BiasAdd/ReadVariableOpReadVariableOp/model_7_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?-*
dtype0?
model_7/dense_7/BiasAddBiasAdd model_7/dense_7/MatMul:product:0.model_7/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????-g
model_7/reshape_3/ShapeShape model_7/dense_7/BiasAdd:output:0*
T0*
_output_shapes
:o
%model_7/reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'model_7/reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'model_7/reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
model_7/reshape_3/strided_sliceStridedSlice model_7/reshape_3/Shape:output:0.model_7/reshape_3/strided_slice/stack:output:00model_7/reshape_3/strided_slice/stack_1:output:00model_7/reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!model_7/reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :c
!model_7/reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :c
!model_7/reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :(?
model_7/reshape_3/Reshape/shapePack(model_7/reshape_3/strided_slice:output:0*model_7/reshape_3/Reshape/shape/1:output:0*model_7/reshape_3/Reshape/shape/2:output:0*model_7/reshape_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
model_7/reshape_3/ReshapeReshape model_7/dense_7/BiasAdd:output:0(model_7/reshape_3/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????(r
 model_7/conv2d_transpose_6/ShapeShape"model_7/reshape_3/Reshape:output:0*
T0*
_output_shapes
:x
.model_7/conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model_7/conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model_7/conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(model_7/conv2d_transpose_6/strided_sliceStridedSlice)model_7/conv2d_transpose_6/Shape:output:07model_7/conv2d_transpose_6/strided_slice/stack:output:09model_7/conv2d_transpose_6/strided_slice/stack_1:output:09model_7/conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"model_7/conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"model_7/conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d
"model_7/conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :(?
 model_7/conv2d_transpose_6/stackPack1model_7/conv2d_transpose_6/strided_slice:output:0+model_7/conv2d_transpose_6/stack/1:output:0+model_7/conv2d_transpose_6/stack/2:output:0+model_7/conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:z
0model_7/conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2model_7/conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2model_7/conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*model_7/conv2d_transpose_6/strided_slice_1StridedSlice)model_7/conv2d_transpose_6/stack:output:09model_7/conv2d_transpose_6/strided_slice_1/stack:output:0;model_7/conv2d_transpose_6/strided_slice_1/stack_1:output:0;model_7/conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
:model_7/conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_7_conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:((*
dtype0?
+model_7/conv2d_transpose_6/conv2d_transposeConv2DBackpropInput)model_7/conv2d_transpose_6/stack:output:0Bmodel_7/conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0"model_7/reshape_3/Reshape:output:0*
T0*/
_output_shapes
:?????????(*
paddingSAME*
strides
?
1model_7/conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp:model_7_conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
"model_7/conv2d_transpose_6/BiasAddBiasAdd4model_7/conv2d_transpose_6/conv2d_transpose:output:09model_7/conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(?
model_7/conv2d_transpose_6/TanhTanh+model_7/conv2d_transpose_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(?
'model_7/conv2d_25/Conv2D/ReadVariableOpReadVariableOp0model_7_conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:((*
dtype0?
model_7/conv2d_25/Conv2DConv2D#model_7/conv2d_transpose_6/Tanh:y:0/model_7/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
paddingSAME*
strides
?
(model_7/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp1model_7_conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
model_7/conv2d_25/BiasAddBiasAdd!model_7/conv2d_25/Conv2D:output:00model_7/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(|
model_7/conv2d_25/TanhTanh"model_7/conv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(j
 model_7/conv2d_transpose_7/ShapeShapemodel_7/conv2d_25/Tanh:y:0*
T0*
_output_shapes
:x
.model_7/conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model_7/conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model_7/conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(model_7/conv2d_transpose_7/strided_sliceStridedSlice)model_7/conv2d_transpose_7/Shape:output:07model_7/conv2d_transpose_7/strided_slice/stack:output:09model_7/conv2d_transpose_7/strided_slice/stack_1:output:09model_7/conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"model_7/conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :0d
"model_7/conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :0d
"model_7/conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
 model_7/conv2d_transpose_7/stackPack1model_7/conv2d_transpose_7/strided_slice:output:0+model_7/conv2d_transpose_7/stack/1:output:0+model_7/conv2d_transpose_7/stack/2:output:0+model_7/conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:z
0model_7/conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2model_7/conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2model_7/conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*model_7/conv2d_transpose_7/strided_slice_1StridedSlice)model_7/conv2d_transpose_7/stack:output:09model_7/conv2d_transpose_7/strided_slice_1/stack:output:0;model_7/conv2d_transpose_7/strided_slice_1/stack_1:output:0;model_7/conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
:model_7/conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_7_conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
:(*
dtype0?
+model_7/conv2d_transpose_7/conv2d_transposeConv2DBackpropInput)model_7/conv2d_transpose_7/stack:output:0Bmodel_7/conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0model_7/conv2d_25/Tanh:y:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
1model_7/conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp:model_7_conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
"model_7/conv2d_transpose_7/BiasAddBiasAdd4model_7/conv2d_transpose_7/conv2d_transpose:output:09model_7/conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
model_7/conv2d_transpose_7/TanhTanh+model_7/conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00?
'model_7/conv2d_26/Conv2D/ReadVariableOpReadVariableOp0model_7_conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
model_7/conv2d_26/Conv2DConv2D#model_7/conv2d_transpose_7/Tanh:y:0/model_7/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
(model_7/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp1model_7_conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_7/conv2d_26/BiasAddBiasAdd!model_7/conv2d_26/Conv2D:output:00model_7/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00|
model_7/conv2d_26/TanhTanh"model_7/conv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00?
'model_7/conv2d_27/Conv2D/ReadVariableOpReadVariableOp0model_7_conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
model_7/conv2d_27/Conv2DConv2Dmodel_7/conv2d_26/Tanh:y:0/model_7/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
(model_7/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp1model_7_conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_7/conv2d_27/BiasAddBiasAdd!model_7/conv2d_27/Conv2D:output:00model_7/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
model_7/conv2d_27/SigmoidSigmoid"model_7/conv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00t
IdentityIdentitymodel_7/conv2d_27/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp)^model_6/conv2d_21/BiasAdd/ReadVariableOp(^model_6/conv2d_21/Conv2D/ReadVariableOp)^model_6/conv2d_22/BiasAdd/ReadVariableOp(^model_6/conv2d_22/Conv2D/ReadVariableOp)^model_6/conv2d_23/BiasAdd/ReadVariableOp(^model_6/conv2d_23/Conv2D/ReadVariableOp)^model_6/conv2d_24/BiasAdd/ReadVariableOp(^model_6/conv2d_24/Conv2D/ReadVariableOp'^model_6/dense_6/BiasAdd/ReadVariableOp&^model_6/dense_6/MatMul/ReadVariableOp)^model_7/conv2d_25/BiasAdd/ReadVariableOp(^model_7/conv2d_25/Conv2D/ReadVariableOp)^model_7/conv2d_26/BiasAdd/ReadVariableOp(^model_7/conv2d_26/Conv2D/ReadVariableOp)^model_7/conv2d_27/BiasAdd/ReadVariableOp(^model_7/conv2d_27/Conv2D/ReadVariableOp2^model_7/conv2d_transpose_6/BiasAdd/ReadVariableOp;^model_7/conv2d_transpose_6/conv2d_transpose/ReadVariableOp2^model_7/conv2d_transpose_7/BiasAdd/ReadVariableOp;^model_7/conv2d_transpose_7/conv2d_transpose/ReadVariableOp'^model_7/dense_7/BiasAdd/ReadVariableOp&^model_7/dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????00: : : : : : : : : : : : : : : : : : : : : : 2T
(model_6/conv2d_21/BiasAdd/ReadVariableOp(model_6/conv2d_21/BiasAdd/ReadVariableOp2R
'model_6/conv2d_21/Conv2D/ReadVariableOp'model_6/conv2d_21/Conv2D/ReadVariableOp2T
(model_6/conv2d_22/BiasAdd/ReadVariableOp(model_6/conv2d_22/BiasAdd/ReadVariableOp2R
'model_6/conv2d_22/Conv2D/ReadVariableOp'model_6/conv2d_22/Conv2D/ReadVariableOp2T
(model_6/conv2d_23/BiasAdd/ReadVariableOp(model_6/conv2d_23/BiasAdd/ReadVariableOp2R
'model_6/conv2d_23/Conv2D/ReadVariableOp'model_6/conv2d_23/Conv2D/ReadVariableOp2T
(model_6/conv2d_24/BiasAdd/ReadVariableOp(model_6/conv2d_24/BiasAdd/ReadVariableOp2R
'model_6/conv2d_24/Conv2D/ReadVariableOp'model_6/conv2d_24/Conv2D/ReadVariableOp2P
&model_6/dense_6/BiasAdd/ReadVariableOp&model_6/dense_6/BiasAdd/ReadVariableOp2N
%model_6/dense_6/MatMul/ReadVariableOp%model_6/dense_6/MatMul/ReadVariableOp2T
(model_7/conv2d_25/BiasAdd/ReadVariableOp(model_7/conv2d_25/BiasAdd/ReadVariableOp2R
'model_7/conv2d_25/Conv2D/ReadVariableOp'model_7/conv2d_25/Conv2D/ReadVariableOp2T
(model_7/conv2d_26/BiasAdd/ReadVariableOp(model_7/conv2d_26/BiasAdd/ReadVariableOp2R
'model_7/conv2d_26/Conv2D/ReadVariableOp'model_7/conv2d_26/Conv2D/ReadVariableOp2T
(model_7/conv2d_27/BiasAdd/ReadVariableOp(model_7/conv2d_27/BiasAdd/ReadVariableOp2R
'model_7/conv2d_27/Conv2D/ReadVariableOp'model_7/conv2d_27/Conv2D/ReadVariableOp2f
1model_7/conv2d_transpose_6/BiasAdd/ReadVariableOp1model_7/conv2d_transpose_6/BiasAdd/ReadVariableOp2x
:model_7/conv2d_transpose_6/conv2d_transpose/ReadVariableOp:model_7/conv2d_transpose_6/conv2d_transpose/ReadVariableOp2f
1model_7/conv2d_transpose_7/BiasAdd/ReadVariableOp1model_7/conv2d_transpose_7/BiasAdd/ReadVariableOp2x
:model_7/conv2d_transpose_7/conv2d_transpose/ReadVariableOp:model_7/conv2d_transpose_7/conv2d_transpose/ReadVariableOp2P
&model_7/dense_7/BiasAdd/ReadVariableOp&model_7/dense_7/BiasAdd/ReadVariableOp2N
%model_7/dense_7/MatMul/ReadVariableOp%model_7/dense_7/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?	
?
C__inference_dense_6_layer_call_and_return_conditional_losses_382592

inputs1
matmul_readvariableop_resource:	?-<-
biasadd_readvariableop_resource:<
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?-<*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????<w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????-: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????-
 
_user_specified_nameinputs
?
?
G__inference_autoencoder_layer_call_and_return_conditional_losses_381309

inputs(
model_6_381262:
model_6_381264:(
model_6_381266:
model_6_381268:(
model_6_381270:(
model_6_381272:((
model_6_381274:((
model_6_381276:(!
model_6_381278:	?-<
model_6_381280:<!
model_7_381283:	<?-
model_7_381285:	?-(
model_7_381287:((
model_7_381289:((
model_7_381291:((
model_7_381293:((
model_7_381295:(
model_7_381297:(
model_7_381299:
model_7_381301:(
model_7_381303:
model_7_381305:
identity??model_6/StatefulPartitionedCall?model_7/StatefulPartitionedCall?
model_6/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_6_381262model_6_381264model_6_381266model_6_381268model_6_381270model_6_381272model_6_381274model_6_381276model_6_381278model_6_381280*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_6_layer_call_and_return_conditional_losses_380503?
model_7/StatefulPartitionedCallStatefulPartitionedCall(model_6/StatefulPartitionedCall:output:0model_7_381283model_7_381285model_7_381287model_7_381289model_7_381291model_7_381293model_7_381295model_7_381297model_7_381299model_7_381301model_7_381303model_7_381305*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_7_layer_call_and_return_conditional_losses_380990
IdentityIdentity(model_7/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp ^model_6/StatefulPartitionedCall ^model_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????00: : : : : : : : : : : : : : : : : : : : : : 2B
model_6/StatefulPartitionedCallmodel_6/StatefulPartitionedCall2B
model_7/StatefulPartitionedCallmodel_7/StatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?$
?
C__inference_model_6_layer_call_and_return_conditional_losses_380503

inputs*
conv2d_21_380408:
conv2d_21_380410:*
conv2d_22_380425:
conv2d_22_380427:*
conv2d_23_380449:(
conv2d_23_380451:(*
conv2d_24_380466:((
conv2d_24_380468:(!
dense_6_380497:	?-<
dense_6_380499:<
identity??!conv2d_21/StatefulPartitionedCall?!conv2d_22/StatefulPartitionedCall?!conv2d_23/StatefulPartitionedCall?!conv2d_24/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_21_380408conv2d_21_380410*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_21_layer_call_and_return_conditional_losses_380407?
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0conv2d_22_380425conv2d_22_380427*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_22_layer_call_and_return_conditional_losses_380424?
dropout_6/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_380435?
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall"dropout_6/PartitionedCall:output:0conv2d_23_380449conv2d_23_380451*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_23_layer_call_and_return_conditional_losses_380448?
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0conv2d_24_380466conv2d_24_380468*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_24_layer_call_and_return_conditional_losses_380465?
dropout_7/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_380476?
flatten_3/PartitionedCallPartitionedCall"dropout_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????-* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_380484?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_380497dense_6_380499*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_380496w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????<?
NoOpNoOp"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????00: : : : : : : : : : 2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?	
?
C__inference_dense_7_layer_call_and_return_conditional_losses_380906

inputs1
matmul_readvariableop_resource:	<?-.
biasadd_readvariableop_resource:	?-
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	<?-*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????-s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?-*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????-`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????-w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?
F
*__inference_dropout_6_layer_call_fn_382473

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_380435h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

d
E__inference_dropout_7_layer_call_and_return_conditional_losses_382562

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????(C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????(*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????(w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????(q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????(a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
(__inference_model_7_layer_call_fn_382237

inputs
unknown:	<?-
	unknown_0:	?-#
	unknown_1:((
	unknown_2:(#
	unknown_3:((
	unknown_4:(#
	unknown_5:(
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_7_layer_call_and_return_conditional_losses_380990w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????<: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?

d
E__inference_dropout_7_layer_call_and_return_conditional_losses_380562

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????(C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????(*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????(w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????(q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????(a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?

?
E__inference_conv2d_22_layer_call_and_return_conditional_losses_382468

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????X
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????_
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????00: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?'
?
C__inference_model_6_layer_call_and_return_conditional_losses_380687

inputs*
conv2d_21_380658:
conv2d_21_380660:*
conv2d_22_380663:
conv2d_22_380665:*
conv2d_23_380669:(
conv2d_23_380671:(*
conv2d_24_380674:((
conv2d_24_380676:(!
dense_6_380681:	?-<
dense_6_380683:<
identity??!conv2d_21/StatefulPartitionedCall?!conv2d_22/StatefulPartitionedCall?!conv2d_23/StatefulPartitionedCall?!conv2d_24/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?!dropout_6/StatefulPartitionedCall?!dropout_7/StatefulPartitionedCall?
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_21_380658conv2d_21_380660*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_21_layer_call_and_return_conditional_losses_380407?
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0conv2d_22_380663conv2d_22_380665*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_22_layer_call_and_return_conditional_losses_380424?
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_380605?
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall*dropout_6/StatefulPartitionedCall:output:0conv2d_23_380669conv2d_23_380671*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_23_layer_call_and_return_conditional_losses_380448?
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0conv2d_24_380674conv2d_24_380676*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_24_layer_call_and_return_conditional_losses_380465?
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_380562?
flatten_3/PartitionedCallPartitionedCall*dropout_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????-* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_380484?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_380681dense_6_380683*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_380496w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????<?
NoOpNoOp"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????00: : : : : : : : : : 2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?

?
(__inference_model_6_layer_call_fn_382110

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:(
	unknown_4:(#
	unknown_5:((
	unknown_6:(
	unknown_7:	?-<
	unknown_8:<
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_6_layer_call_and_return_conditional_losses_380687o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????00: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
*__inference_conv2d_25_layer_call_fn_382682

inputs!
unknown:((
	unknown_0:(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_25_layer_call_and_return_conditional_losses_380944w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?

?
(__inference_model_6_layer_call_fn_380526
encoder_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:(
	unknown_4:(#
	unknown_5:((
	unknown_6:(
	unknown_7:	?-<
	unknown_8:<
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_6_layer_call_and_return_conditional_losses_380503o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????00: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:?????????00
'
_user_specified_nameencoder_input
?
?
G__inference_autoencoder_layer_call_and_return_conditional_losses_381653
encoder_input(
model_6_381606:
model_6_381608:(
model_6_381610:
model_6_381612:(
model_6_381614:(
model_6_381616:((
model_6_381618:((
model_6_381620:(!
model_6_381622:	?-<
model_6_381624:<!
model_7_381627:	<?-
model_7_381629:	?-(
model_7_381631:((
model_7_381633:((
model_7_381635:((
model_7_381637:((
model_7_381639:(
model_7_381641:(
model_7_381643:
model_7_381645:(
model_7_381647:
model_7_381649:
identity??model_6/StatefulPartitionedCall?model_7/StatefulPartitionedCall?
model_6/StatefulPartitionedCallStatefulPartitionedCallencoder_inputmodel_6_381606model_6_381608model_6_381610model_6_381612model_6_381614model_6_381616model_6_381618model_6_381620model_6_381622model_6_381624*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_6_layer_call_and_return_conditional_losses_380687?
model_7/StatefulPartitionedCallStatefulPartitionedCall(model_6/StatefulPartitionedCall:output:0model_7_381627model_7_381629model_7_381631model_7_381633model_7_381635model_7_381637model_7_381639model_7_381641model_7_381643model_7_381645model_7_381647model_7_381649*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_7_layer_call_and_return_conditional_losses_381129
IdentityIdentity(model_7/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp ^model_6/StatefulPartitionedCall ^model_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????00: : : : : : : : : : : : : : : : : : : : : : 2B
model_6/StatefulPartitionedCallmodel_6/StatefulPartitionedCall2B
model_7/StatefulPartitionedCallmodel_7/StatefulPartitionedCall:^ Z
/
_output_shapes
:?????????00
'
_user_specified_nameencoder_input
?
?
(__inference_model_7_layer_call_fn_381017
input_4
unknown:	<?-
	unknown_0:	?-#
	unknown_1:((
	unknown_2:(#
	unknown_3:((
	unknown_4:(#
	unknown_5:(
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_7_layer_call_and_return_conditional_losses_380990w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????<: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????<
!
_user_specified_name	input_4
?
F
*__inference_dropout_7_layer_call_fn_382540

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_380476h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
,__inference_autoencoder_layer_call_fn_381757

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:(
	unknown_4:(#
	unknown_5:((
	unknown_6:(
	unknown_7:	?-<
	unknown_8:<
	unknown_9:	<?-

unknown_10:	?-$

unknown_11:((

unknown_12:($

unknown_13:((

unknown_14:($

unknown_15:(

unknown_16:$

unknown_17:

unknown_18:$

unknown_19:

unknown_20:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_autoencoder_layer_call_and_return_conditional_losses_381457w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????00: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?

?
E__inference_conv2d_21_layer_call_and_return_conditional_losses_382448

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00X
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????00_
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:?????????00w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????00: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?e
?

C__inference_model_7_layer_call_and_return_conditional_losses_382347

inputs9
&dense_7_matmul_readvariableop_resource:	<?-6
'dense_7_biasadd_readvariableop_resource:	?-U
;conv2d_transpose_6_conv2d_transpose_readvariableop_resource:((@
2conv2d_transpose_6_biasadd_readvariableop_resource:(B
(conv2d_25_conv2d_readvariableop_resource:((7
)conv2d_25_biasadd_readvariableop_resource:(U
;conv2d_transpose_7_conv2d_transpose_readvariableop_resource:(@
2conv2d_transpose_7_biasadd_readvariableop_resource:B
(conv2d_26_conv2d_readvariableop_resource:7
)conv2d_26_biasadd_readvariableop_resource:B
(conv2d_27_conv2d_readvariableop_resource:7
)conv2d_27_biasadd_readvariableop_resource:
identity?? conv2d_25/BiasAdd/ReadVariableOp?conv2d_25/Conv2D/ReadVariableOp? conv2d_26/BiasAdd/ReadVariableOp?conv2d_26/Conv2D/ReadVariableOp? conv2d_27/BiasAdd/ReadVariableOp?conv2d_27/Conv2D/ReadVariableOp?)conv2d_transpose_6/BiasAdd/ReadVariableOp?2conv2d_transpose_6/conv2d_transpose/ReadVariableOp?)conv2d_transpose_7/BiasAdd/ReadVariableOp?2conv2d_transpose_7/conv2d_transpose/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	<?-*
dtype0z
dense_7/MatMulMatMulinputs%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????-?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?-*
dtype0?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????-W
reshape_3/ShapeShapedense_7/BiasAdd:output:0*
T0*
_output_shapes
:g
reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape_3/strided_sliceStridedSlicereshape_3/Shape:output:0&reshape_3/strided_slice/stack:output:0(reshape_3/strided_slice/stack_1:output:0(reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :(?
reshape_3/Reshape/shapePack reshape_3/strided_slice:output:0"reshape_3/Reshape/shape/1:output:0"reshape_3/Reshape/shape/2:output:0"reshape_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
reshape_3/ReshapeReshapedense_7/BiasAdd:output:0 reshape_3/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????(b
conv2d_transpose_6/ShapeShapereshape_3/Reshape:output:0*
T0*
_output_shapes
:p
&conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_6/strided_sliceStridedSlice!conv2d_transpose_6/Shape:output:0/conv2d_transpose_6/strided_slice/stack:output:01conv2d_transpose_6/strided_slice/stack_1:output:01conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :(?
conv2d_transpose_6/stackPack)conv2d_transpose_6/strided_slice:output:0#conv2d_transpose_6/stack/1:output:0#conv2d_transpose_6/stack/2:output:0#conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_6/strided_slice_1StridedSlice!conv2d_transpose_6/stack:output:01conv2d_transpose_6/strided_slice_1/stack:output:03conv2d_transpose_6/strided_slice_1/stack_1:output:03conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:((*
dtype0?
#conv2d_transpose_6/conv2d_transposeConv2DBackpropInput!conv2d_transpose_6/stack:output:0:conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0reshape_3/Reshape:output:0*
T0*/
_output_shapes
:?????????(*
paddingSAME*
strides
?
)conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
conv2d_transpose_6/BiasAddBiasAdd,conv2d_transpose_6/conv2d_transpose:output:01conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(~
conv2d_transpose_6/TanhTanh#conv2d_transpose_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(?
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:((*
dtype0?
conv2d_25/Conv2DConv2Dconv2d_transpose_6/Tanh:y:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
paddingSAME*
strides
?
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(l
conv2d_25/TanhTanhconv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(Z
conv2d_transpose_7/ShapeShapeconv2d_25/Tanh:y:0*
T0*
_output_shapes
:p
&conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_7/strided_sliceStridedSlice!conv2d_transpose_7/Shape:output:0/conv2d_transpose_7/strided_slice/stack:output:01conv2d_transpose_7/strided_slice/stack_1:output:01conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :0\
conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :0\
conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_7/stackPack)conv2d_transpose_7/strided_slice:output:0#conv2d_transpose_7/stack/1:output:0#conv2d_transpose_7/stack/2:output:0#conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_7/strided_slice_1StridedSlice!conv2d_transpose_7/stack:output:01conv2d_transpose_7/strided_slice_1/stack:output:03conv2d_transpose_7/strided_slice_1/stack_1:output:03conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
:(*
dtype0?
#conv2d_transpose_7/conv2d_transposeConv2DBackpropInput!conv2d_transpose_7/stack:output:0:conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0conv2d_25/Tanh:y:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
)conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_7/BiasAddBiasAdd,conv2d_transpose_7/conv2d_transpose:output:01conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00~
conv2d_transpose_7/TanhTanh#conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00?
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_26/Conv2DConv2Dconv2d_transpose_7/Tanh:y:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00l
conv2d_26/TanhTanhconv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00?
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_27/Conv2DConv2Dconv2d_26/Tanh:y:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00r
conv2d_27/SigmoidSigmoidconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00l
IdentityIdentityconv2d_27/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp*^conv2d_transpose_6/BiasAdd/ReadVariableOp3^conv2d_transpose_6/conv2d_transpose/ReadVariableOp*^conv2d_transpose_7/BiasAdd/ReadVariableOp3^conv2d_transpose_7/conv2d_transpose/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????<: : : : : : : : : : : : 2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2V
)conv2d_transpose_6/BiasAdd/ReadVariableOp)conv2d_transpose_6/BiasAdd/ReadVariableOp2h
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_7/BiasAdd/ReadVariableOp)conv2d_transpose_7/BiasAdd/ReadVariableOp2h
2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?

?
E__inference_conv2d_23_layer_call_and_return_conditional_losses_380448

inputs8
conv2d_readvariableop_resource:(-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:(*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(X
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????(_
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:?????????(w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_conv2d_23_layer_call_fn_382504

inputs!
unknown:(
	unknown_0:(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_23_layer_call_and_return_conditional_losses_380448w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_conv2d_26_layer_call_fn_382745

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_26_layer_call_and_return_conditional_losses_380966w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????00: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?%
?
C__inference_model_7_layer_call_and_return_conditional_losses_381255
input_4!
dense_7_381223:	<?-
dense_7_381225:	?-3
conv2d_transpose_6_381229:(('
conv2d_transpose_6_381231:(*
conv2d_25_381234:((
conv2d_25_381236:(3
conv2d_transpose_7_381239:('
conv2d_transpose_7_381241:*
conv2d_26_381244:
conv2d_26_381246:*
conv2d_27_381249:
conv2d_27_381251:
identity??!conv2d_25/StatefulPartitionedCall?!conv2d_26/StatefulPartitionedCall?!conv2d_27/StatefulPartitionedCall?*conv2d_transpose_6/StatefulPartitionedCall?*conv2d_transpose_7/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_7_381223dense_7_381225*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_380906?
reshape_3/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_reshape_3_layer_call_and_return_conditional_losses_380926?
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall"reshape_3/PartitionedCall:output:0conv2d_transpose_6_381229conv2d_transpose_6_381231*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_380837?
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0conv2d_25_381234conv2d_25_381236*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_25_layer_call_and_return_conditional_losses_380944?
*conv2d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0conv2d_transpose_7_381239conv2d_transpose_7_381241*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_380882?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_7/StatefulPartitionedCall:output:0conv2d_26_381244conv2d_26_381246*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_26_layer_call_and_return_conditional_losses_380966?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0conv2d_27_381249conv2d_27_381251*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_27_layer_call_and_return_conditional_losses_380983?
IdentityIdentity*conv2d_27/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall+^conv2d_transpose_6/StatefulPartitionedCall+^conv2d_transpose_7/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????<: : : : : : : : : : : : 2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2X
*conv2d_transpose_7/StatefulPartitionedCall*conv2d_transpose_7/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:P L
'
_output_shapes
:?????????<
!
_user_specified_name	input_4
?
?
$__inference_signature_wrapper_382060
encoder_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:(
	unknown_4:(#
	unknown_5:((
	unknown_6:(
	unknown_7:	?-<
	unknown_8:<
	unknown_9:	<?-

unknown_10:	?-$

unknown_11:((

unknown_12:($

unknown_13:((

unknown_14:($

unknown_15:(

unknown_16:$

unknown_17:

unknown_18:$

unknown_19:

unknown_20:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_380389w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????00: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:?????????00
'
_user_specified_nameencoder_input
?
?
,__inference_autoencoder_layer_call_fn_381553
encoder_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:(
	unknown_4:(#
	unknown_5:((
	unknown_6:(
	unknown_7:	?-<
	unknown_8:<
	unknown_9:	<?-

unknown_10:	?-$

unknown_11:((

unknown_12:($

unknown_13:((

unknown_14:($

unknown_15:(

unknown_16:$

unknown_17:

unknown_18:$

unknown_19:

unknown_20:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_autoencoder_layer_call_and_return_conditional_losses_381457w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????00: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:?????????00
'
_user_specified_nameencoder_input
?

d
E__inference_dropout_6_layer_call_and_return_conditional_losses_382495

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
*__inference_dropout_6_layer_call_fn_382478

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_380605w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_flatten_3_layer_call_fn_382567

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????-* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_380484a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????-"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
*__inference_conv2d_21_layer_call_fn_382437

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_21_layer_call_and_return_conditional_losses_380407w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????00: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?!
?
N__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_382736

inputsB
(conv2d_transpose_readvariableop_resource:(-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:(*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????j
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????q
IdentityIdentityTanh:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????(
 
_user_specified_nameinputs
?
?
E__inference_conv2d_27_layer_call_and_return_conditional_losses_380983

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00^
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:?????????00b
IdentityIdentitySigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????00w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????00: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
E__inference_conv2d_27_layer_call_and_return_conditional_losses_382776

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00^
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:?????????00b
IdentityIdentitySigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????00w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????00: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?!
?
N__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_380837

inputsB
(conv2d_transpose_readvariableop_resource:((-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :(y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:((*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????(*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????(j
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????(q
IdentityIdentityTanh:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????(?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????(
 
_user_specified_nameinputs
ب
?
G__inference_autoencoder_layer_call_and_return_conditional_losses_381876

inputsJ
0model_6_conv2d_21_conv2d_readvariableop_resource:?
1model_6_conv2d_21_biasadd_readvariableop_resource:J
0model_6_conv2d_22_conv2d_readvariableop_resource:?
1model_6_conv2d_22_biasadd_readvariableop_resource:J
0model_6_conv2d_23_conv2d_readvariableop_resource:(?
1model_6_conv2d_23_biasadd_readvariableop_resource:(J
0model_6_conv2d_24_conv2d_readvariableop_resource:((?
1model_6_conv2d_24_biasadd_readvariableop_resource:(A
.model_6_dense_6_matmul_readvariableop_resource:	?-<=
/model_6_dense_6_biasadd_readvariableop_resource:<A
.model_7_dense_7_matmul_readvariableop_resource:	<?->
/model_7_dense_7_biasadd_readvariableop_resource:	?-]
Cmodel_7_conv2d_transpose_6_conv2d_transpose_readvariableop_resource:((H
:model_7_conv2d_transpose_6_biasadd_readvariableop_resource:(J
0model_7_conv2d_25_conv2d_readvariableop_resource:((?
1model_7_conv2d_25_biasadd_readvariableop_resource:(]
Cmodel_7_conv2d_transpose_7_conv2d_transpose_readvariableop_resource:(H
:model_7_conv2d_transpose_7_biasadd_readvariableop_resource:J
0model_7_conv2d_26_conv2d_readvariableop_resource:?
1model_7_conv2d_26_biasadd_readvariableop_resource:J
0model_7_conv2d_27_conv2d_readvariableop_resource:?
1model_7_conv2d_27_biasadd_readvariableop_resource:
identity??(model_6/conv2d_21/BiasAdd/ReadVariableOp?'model_6/conv2d_21/Conv2D/ReadVariableOp?(model_6/conv2d_22/BiasAdd/ReadVariableOp?'model_6/conv2d_22/Conv2D/ReadVariableOp?(model_6/conv2d_23/BiasAdd/ReadVariableOp?'model_6/conv2d_23/Conv2D/ReadVariableOp?(model_6/conv2d_24/BiasAdd/ReadVariableOp?'model_6/conv2d_24/Conv2D/ReadVariableOp?&model_6/dense_6/BiasAdd/ReadVariableOp?%model_6/dense_6/MatMul/ReadVariableOp?(model_7/conv2d_25/BiasAdd/ReadVariableOp?'model_7/conv2d_25/Conv2D/ReadVariableOp?(model_7/conv2d_26/BiasAdd/ReadVariableOp?'model_7/conv2d_26/Conv2D/ReadVariableOp?(model_7/conv2d_27/BiasAdd/ReadVariableOp?'model_7/conv2d_27/Conv2D/ReadVariableOp?1model_7/conv2d_transpose_6/BiasAdd/ReadVariableOp?:model_7/conv2d_transpose_6/conv2d_transpose/ReadVariableOp?1model_7/conv2d_transpose_7/BiasAdd/ReadVariableOp?:model_7/conv2d_transpose_7/conv2d_transpose/ReadVariableOp?&model_7/dense_7/BiasAdd/ReadVariableOp?%model_7/dense_7/MatMul/ReadVariableOp?
'model_6/conv2d_21/Conv2D/ReadVariableOpReadVariableOp0model_6_conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
model_6/conv2d_21/Conv2DConv2Dinputs/model_6/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
(model_6/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp1model_6_conv2d_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_6/conv2d_21/BiasAddBiasAdd!model_6/conv2d_21/Conv2D:output:00model_6/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00|
model_6/conv2d_21/TanhTanh"model_6/conv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00?
'model_6/conv2d_22/Conv2D/ReadVariableOpReadVariableOp0model_6_conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
model_6/conv2d_22/Conv2DConv2Dmodel_6/conv2d_21/Tanh:y:0/model_6/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
(model_6/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp1model_6_conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_6/conv2d_22/BiasAddBiasAdd!model_6/conv2d_22/Conv2D:output:00model_6/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????|
model_6/conv2d_22/TanhTanh"model_6/conv2d_22/BiasAdd:output:0*
T0*/
_output_shapes
:?????????|
model_6/dropout_6/IdentityIdentitymodel_6/conv2d_22/Tanh:y:0*
T0*/
_output_shapes
:??????????
'model_6/conv2d_23/Conv2D/ReadVariableOpReadVariableOp0model_6_conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype0?
model_6/conv2d_23/Conv2DConv2D#model_6/dropout_6/Identity:output:0/model_6/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
paddingSAME*
strides
?
(model_6/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp1model_6_conv2d_23_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
model_6/conv2d_23/BiasAddBiasAdd!model_6/conv2d_23/Conv2D:output:00model_6/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(|
model_6/conv2d_23/TanhTanh"model_6/conv2d_23/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(?
'model_6/conv2d_24/Conv2D/ReadVariableOpReadVariableOp0model_6_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
:((*
dtype0?
model_6/conv2d_24/Conv2DConv2Dmodel_6/conv2d_23/Tanh:y:0/model_6/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
paddingSAME*
strides
?
(model_6/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp1model_6_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
model_6/conv2d_24/BiasAddBiasAdd!model_6/conv2d_24/Conv2D:output:00model_6/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(|
model_6/conv2d_24/TanhTanh"model_6/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(|
model_6/dropout_7/IdentityIdentitymodel_6/conv2d_24/Tanh:y:0*
T0*/
_output_shapes
:?????????(h
model_6/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
model_6/flatten_3/ReshapeReshape#model_6/dropout_7/Identity:output:0 model_6/flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????-?
%model_6/dense_6/MatMul/ReadVariableOpReadVariableOp.model_6_dense_6_matmul_readvariableop_resource*
_output_shapes
:	?-<*
dtype0?
model_6/dense_6/MatMulMatMul"model_6/flatten_3/Reshape:output:0-model_6/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<?
&model_6/dense_6/BiasAdd/ReadVariableOpReadVariableOp/model_6_dense_6_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0?
model_6/dense_6/BiasAddBiasAdd model_6/dense_6/MatMul:product:0.model_6/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<?
%model_7/dense_7/MatMul/ReadVariableOpReadVariableOp.model_7_dense_7_matmul_readvariableop_resource*
_output_shapes
:	<?-*
dtype0?
model_7/dense_7/MatMulMatMul model_6/dense_6/BiasAdd:output:0-model_7/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????-?
&model_7/dense_7/BiasAdd/ReadVariableOpReadVariableOp/model_7_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?-*
dtype0?
model_7/dense_7/BiasAddBiasAdd model_7/dense_7/MatMul:product:0.model_7/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????-g
model_7/reshape_3/ShapeShape model_7/dense_7/BiasAdd:output:0*
T0*
_output_shapes
:o
%model_7/reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'model_7/reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'model_7/reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
model_7/reshape_3/strided_sliceStridedSlice model_7/reshape_3/Shape:output:0.model_7/reshape_3/strided_slice/stack:output:00model_7/reshape_3/strided_slice/stack_1:output:00model_7/reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!model_7/reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :c
!model_7/reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :c
!model_7/reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :(?
model_7/reshape_3/Reshape/shapePack(model_7/reshape_3/strided_slice:output:0*model_7/reshape_3/Reshape/shape/1:output:0*model_7/reshape_3/Reshape/shape/2:output:0*model_7/reshape_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
model_7/reshape_3/ReshapeReshape model_7/dense_7/BiasAdd:output:0(model_7/reshape_3/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????(r
 model_7/conv2d_transpose_6/ShapeShape"model_7/reshape_3/Reshape:output:0*
T0*
_output_shapes
:x
.model_7/conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model_7/conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model_7/conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(model_7/conv2d_transpose_6/strided_sliceStridedSlice)model_7/conv2d_transpose_6/Shape:output:07model_7/conv2d_transpose_6/strided_slice/stack:output:09model_7/conv2d_transpose_6/strided_slice/stack_1:output:09model_7/conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"model_7/conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"model_7/conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d
"model_7/conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :(?
 model_7/conv2d_transpose_6/stackPack1model_7/conv2d_transpose_6/strided_slice:output:0+model_7/conv2d_transpose_6/stack/1:output:0+model_7/conv2d_transpose_6/stack/2:output:0+model_7/conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:z
0model_7/conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2model_7/conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2model_7/conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*model_7/conv2d_transpose_6/strided_slice_1StridedSlice)model_7/conv2d_transpose_6/stack:output:09model_7/conv2d_transpose_6/strided_slice_1/stack:output:0;model_7/conv2d_transpose_6/strided_slice_1/stack_1:output:0;model_7/conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
:model_7/conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_7_conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:((*
dtype0?
+model_7/conv2d_transpose_6/conv2d_transposeConv2DBackpropInput)model_7/conv2d_transpose_6/stack:output:0Bmodel_7/conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0"model_7/reshape_3/Reshape:output:0*
T0*/
_output_shapes
:?????????(*
paddingSAME*
strides
?
1model_7/conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp:model_7_conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
"model_7/conv2d_transpose_6/BiasAddBiasAdd4model_7/conv2d_transpose_6/conv2d_transpose:output:09model_7/conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(?
model_7/conv2d_transpose_6/TanhTanh+model_7/conv2d_transpose_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(?
'model_7/conv2d_25/Conv2D/ReadVariableOpReadVariableOp0model_7_conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:((*
dtype0?
model_7/conv2d_25/Conv2DConv2D#model_7/conv2d_transpose_6/Tanh:y:0/model_7/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
paddingSAME*
strides
?
(model_7/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp1model_7_conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
model_7/conv2d_25/BiasAddBiasAdd!model_7/conv2d_25/Conv2D:output:00model_7/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(|
model_7/conv2d_25/TanhTanh"model_7/conv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(j
 model_7/conv2d_transpose_7/ShapeShapemodel_7/conv2d_25/Tanh:y:0*
T0*
_output_shapes
:x
.model_7/conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model_7/conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model_7/conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(model_7/conv2d_transpose_7/strided_sliceStridedSlice)model_7/conv2d_transpose_7/Shape:output:07model_7/conv2d_transpose_7/strided_slice/stack:output:09model_7/conv2d_transpose_7/strided_slice/stack_1:output:09model_7/conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"model_7/conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :0d
"model_7/conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :0d
"model_7/conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
 model_7/conv2d_transpose_7/stackPack1model_7/conv2d_transpose_7/strided_slice:output:0+model_7/conv2d_transpose_7/stack/1:output:0+model_7/conv2d_transpose_7/stack/2:output:0+model_7/conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:z
0model_7/conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2model_7/conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2model_7/conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*model_7/conv2d_transpose_7/strided_slice_1StridedSlice)model_7/conv2d_transpose_7/stack:output:09model_7/conv2d_transpose_7/strided_slice_1/stack:output:0;model_7/conv2d_transpose_7/strided_slice_1/stack_1:output:0;model_7/conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
:model_7/conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_7_conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
:(*
dtype0?
+model_7/conv2d_transpose_7/conv2d_transposeConv2DBackpropInput)model_7/conv2d_transpose_7/stack:output:0Bmodel_7/conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0model_7/conv2d_25/Tanh:y:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
1model_7/conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp:model_7_conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
"model_7/conv2d_transpose_7/BiasAddBiasAdd4model_7/conv2d_transpose_7/conv2d_transpose:output:09model_7/conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
model_7/conv2d_transpose_7/TanhTanh+model_7/conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00?
'model_7/conv2d_26/Conv2D/ReadVariableOpReadVariableOp0model_7_conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
model_7/conv2d_26/Conv2DConv2D#model_7/conv2d_transpose_7/Tanh:y:0/model_7/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
(model_7/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp1model_7_conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_7/conv2d_26/BiasAddBiasAdd!model_7/conv2d_26/Conv2D:output:00model_7/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00|
model_7/conv2d_26/TanhTanh"model_7/conv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00?
'model_7/conv2d_27/Conv2D/ReadVariableOpReadVariableOp0model_7_conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
model_7/conv2d_27/Conv2DConv2Dmodel_7/conv2d_26/Tanh:y:0/model_7/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
(model_7/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp1model_7_conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_7/conv2d_27/BiasAddBiasAdd!model_7/conv2d_27/Conv2D:output:00model_7/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
model_7/conv2d_27/SigmoidSigmoid"model_7/conv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00t
IdentityIdentitymodel_7/conv2d_27/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp)^model_6/conv2d_21/BiasAdd/ReadVariableOp(^model_6/conv2d_21/Conv2D/ReadVariableOp)^model_6/conv2d_22/BiasAdd/ReadVariableOp(^model_6/conv2d_22/Conv2D/ReadVariableOp)^model_6/conv2d_23/BiasAdd/ReadVariableOp(^model_6/conv2d_23/Conv2D/ReadVariableOp)^model_6/conv2d_24/BiasAdd/ReadVariableOp(^model_6/conv2d_24/Conv2D/ReadVariableOp'^model_6/dense_6/BiasAdd/ReadVariableOp&^model_6/dense_6/MatMul/ReadVariableOp)^model_7/conv2d_25/BiasAdd/ReadVariableOp(^model_7/conv2d_25/Conv2D/ReadVariableOp)^model_7/conv2d_26/BiasAdd/ReadVariableOp(^model_7/conv2d_26/Conv2D/ReadVariableOp)^model_7/conv2d_27/BiasAdd/ReadVariableOp(^model_7/conv2d_27/Conv2D/ReadVariableOp2^model_7/conv2d_transpose_6/BiasAdd/ReadVariableOp;^model_7/conv2d_transpose_6/conv2d_transpose/ReadVariableOp2^model_7/conv2d_transpose_7/BiasAdd/ReadVariableOp;^model_7/conv2d_transpose_7/conv2d_transpose/ReadVariableOp'^model_7/dense_7/BiasAdd/ReadVariableOp&^model_7/dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????00: : : : : : : : : : : : : : : : : : : : : : 2T
(model_6/conv2d_21/BiasAdd/ReadVariableOp(model_6/conv2d_21/BiasAdd/ReadVariableOp2R
'model_6/conv2d_21/Conv2D/ReadVariableOp'model_6/conv2d_21/Conv2D/ReadVariableOp2T
(model_6/conv2d_22/BiasAdd/ReadVariableOp(model_6/conv2d_22/BiasAdd/ReadVariableOp2R
'model_6/conv2d_22/Conv2D/ReadVariableOp'model_6/conv2d_22/Conv2D/ReadVariableOp2T
(model_6/conv2d_23/BiasAdd/ReadVariableOp(model_6/conv2d_23/BiasAdd/ReadVariableOp2R
'model_6/conv2d_23/Conv2D/ReadVariableOp'model_6/conv2d_23/Conv2D/ReadVariableOp2T
(model_6/conv2d_24/BiasAdd/ReadVariableOp(model_6/conv2d_24/BiasAdd/ReadVariableOp2R
'model_6/conv2d_24/Conv2D/ReadVariableOp'model_6/conv2d_24/Conv2D/ReadVariableOp2P
&model_6/dense_6/BiasAdd/ReadVariableOp&model_6/dense_6/BiasAdd/ReadVariableOp2N
%model_6/dense_6/MatMul/ReadVariableOp%model_6/dense_6/MatMul/ReadVariableOp2T
(model_7/conv2d_25/BiasAdd/ReadVariableOp(model_7/conv2d_25/BiasAdd/ReadVariableOp2R
'model_7/conv2d_25/Conv2D/ReadVariableOp'model_7/conv2d_25/Conv2D/ReadVariableOp2T
(model_7/conv2d_26/BiasAdd/ReadVariableOp(model_7/conv2d_26/BiasAdd/ReadVariableOp2R
'model_7/conv2d_26/Conv2D/ReadVariableOp'model_7/conv2d_26/Conv2D/ReadVariableOp2T
(model_7/conv2d_27/BiasAdd/ReadVariableOp(model_7/conv2d_27/BiasAdd/ReadVariableOp2R
'model_7/conv2d_27/Conv2D/ReadVariableOp'model_7/conv2d_27/Conv2D/ReadVariableOp2f
1model_7/conv2d_transpose_6/BiasAdd/ReadVariableOp1model_7/conv2d_transpose_6/BiasAdd/ReadVariableOp2x
:model_7/conv2d_transpose_6/conv2d_transpose/ReadVariableOp:model_7/conv2d_transpose_6/conv2d_transpose/ReadVariableOp2f
1model_7/conv2d_transpose_7/BiasAdd/ReadVariableOp1model_7/conv2d_transpose_7/BiasAdd/ReadVariableOp2x
:model_7/conv2d_transpose_7/conv2d_transpose/ReadVariableOp:model_7/conv2d_transpose_7/conv2d_transpose/ReadVariableOp2P
&model_7/dense_7/BiasAdd/ReadVariableOp&model_7/dense_7/BiasAdd/ReadVariableOp2N
%model_7/dense_7/MatMul/ReadVariableOp%model_7/dense_7/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
*__inference_conv2d_27_layer_call_fn_382765

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_27_layer_call_and_return_conditional_losses_380983w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????00: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
??
?
!__inference__wrapped_model_380389
encoder_inputV
<autoencoder_model_6_conv2d_21_conv2d_readvariableop_resource:K
=autoencoder_model_6_conv2d_21_biasadd_readvariableop_resource:V
<autoencoder_model_6_conv2d_22_conv2d_readvariableop_resource:K
=autoencoder_model_6_conv2d_22_biasadd_readvariableop_resource:V
<autoencoder_model_6_conv2d_23_conv2d_readvariableop_resource:(K
=autoencoder_model_6_conv2d_23_biasadd_readvariableop_resource:(V
<autoencoder_model_6_conv2d_24_conv2d_readvariableop_resource:((K
=autoencoder_model_6_conv2d_24_biasadd_readvariableop_resource:(M
:autoencoder_model_6_dense_6_matmul_readvariableop_resource:	?-<I
;autoencoder_model_6_dense_6_biasadd_readvariableop_resource:<M
:autoencoder_model_7_dense_7_matmul_readvariableop_resource:	<?-J
;autoencoder_model_7_dense_7_biasadd_readvariableop_resource:	?-i
Oautoencoder_model_7_conv2d_transpose_6_conv2d_transpose_readvariableop_resource:((T
Fautoencoder_model_7_conv2d_transpose_6_biasadd_readvariableop_resource:(V
<autoencoder_model_7_conv2d_25_conv2d_readvariableop_resource:((K
=autoencoder_model_7_conv2d_25_biasadd_readvariableop_resource:(i
Oautoencoder_model_7_conv2d_transpose_7_conv2d_transpose_readvariableop_resource:(T
Fautoencoder_model_7_conv2d_transpose_7_biasadd_readvariableop_resource:V
<autoencoder_model_7_conv2d_26_conv2d_readvariableop_resource:K
=autoencoder_model_7_conv2d_26_biasadd_readvariableop_resource:V
<autoencoder_model_7_conv2d_27_conv2d_readvariableop_resource:K
=autoencoder_model_7_conv2d_27_biasadd_readvariableop_resource:
identity??4autoencoder/model_6/conv2d_21/BiasAdd/ReadVariableOp?3autoencoder/model_6/conv2d_21/Conv2D/ReadVariableOp?4autoencoder/model_6/conv2d_22/BiasAdd/ReadVariableOp?3autoencoder/model_6/conv2d_22/Conv2D/ReadVariableOp?4autoencoder/model_6/conv2d_23/BiasAdd/ReadVariableOp?3autoencoder/model_6/conv2d_23/Conv2D/ReadVariableOp?4autoencoder/model_6/conv2d_24/BiasAdd/ReadVariableOp?3autoencoder/model_6/conv2d_24/Conv2D/ReadVariableOp?2autoencoder/model_6/dense_6/BiasAdd/ReadVariableOp?1autoencoder/model_6/dense_6/MatMul/ReadVariableOp?4autoencoder/model_7/conv2d_25/BiasAdd/ReadVariableOp?3autoencoder/model_7/conv2d_25/Conv2D/ReadVariableOp?4autoencoder/model_7/conv2d_26/BiasAdd/ReadVariableOp?3autoencoder/model_7/conv2d_26/Conv2D/ReadVariableOp?4autoencoder/model_7/conv2d_27/BiasAdd/ReadVariableOp?3autoencoder/model_7/conv2d_27/Conv2D/ReadVariableOp?=autoencoder/model_7/conv2d_transpose_6/BiasAdd/ReadVariableOp?Fautoencoder/model_7/conv2d_transpose_6/conv2d_transpose/ReadVariableOp?=autoencoder/model_7/conv2d_transpose_7/BiasAdd/ReadVariableOp?Fautoencoder/model_7/conv2d_transpose_7/conv2d_transpose/ReadVariableOp?2autoencoder/model_7/dense_7/BiasAdd/ReadVariableOp?1autoencoder/model_7/dense_7/MatMul/ReadVariableOp?
3autoencoder/model_6/conv2d_21/Conv2D/ReadVariableOpReadVariableOp<autoencoder_model_6_conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
$autoencoder/model_6/conv2d_21/Conv2DConv2Dencoder_input;autoencoder/model_6/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
4autoencoder/model_6/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp=autoencoder_model_6_conv2d_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
%autoencoder/model_6/conv2d_21/BiasAddBiasAdd-autoencoder/model_6/conv2d_21/Conv2D:output:0<autoencoder/model_6/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
"autoencoder/model_6/conv2d_21/TanhTanh.autoencoder/model_6/conv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00?
3autoencoder/model_6/conv2d_22/Conv2D/ReadVariableOpReadVariableOp<autoencoder_model_6_conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
$autoencoder/model_6/conv2d_22/Conv2DConv2D&autoencoder/model_6/conv2d_21/Tanh:y:0;autoencoder/model_6/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
4autoencoder/model_6/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp=autoencoder_model_6_conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
%autoencoder/model_6/conv2d_22/BiasAddBiasAdd-autoencoder/model_6/conv2d_22/Conv2D:output:0<autoencoder/model_6/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
"autoencoder/model_6/conv2d_22/TanhTanh.autoencoder/model_6/conv2d_22/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
&autoencoder/model_6/dropout_6/IdentityIdentity&autoencoder/model_6/conv2d_22/Tanh:y:0*
T0*/
_output_shapes
:??????????
3autoencoder/model_6/conv2d_23/Conv2D/ReadVariableOpReadVariableOp<autoencoder_model_6_conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype0?
$autoencoder/model_6/conv2d_23/Conv2DConv2D/autoencoder/model_6/dropout_6/Identity:output:0;autoencoder/model_6/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
paddingSAME*
strides
?
4autoencoder/model_6/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp=autoencoder_model_6_conv2d_23_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
%autoencoder/model_6/conv2d_23/BiasAddBiasAdd-autoencoder/model_6/conv2d_23/Conv2D:output:0<autoencoder/model_6/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(?
"autoencoder/model_6/conv2d_23/TanhTanh.autoencoder/model_6/conv2d_23/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(?
3autoencoder/model_6/conv2d_24/Conv2D/ReadVariableOpReadVariableOp<autoencoder_model_6_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
:((*
dtype0?
$autoencoder/model_6/conv2d_24/Conv2DConv2D&autoencoder/model_6/conv2d_23/Tanh:y:0;autoencoder/model_6/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
paddingSAME*
strides
?
4autoencoder/model_6/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp=autoencoder_model_6_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
%autoencoder/model_6/conv2d_24/BiasAddBiasAdd-autoencoder/model_6/conv2d_24/Conv2D:output:0<autoencoder/model_6/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(?
"autoencoder/model_6/conv2d_24/TanhTanh.autoencoder/model_6/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(?
&autoencoder/model_6/dropout_7/IdentityIdentity&autoencoder/model_6/conv2d_24/Tanh:y:0*
T0*/
_output_shapes
:?????????(t
#autoencoder/model_6/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
%autoencoder/model_6/flatten_3/ReshapeReshape/autoencoder/model_6/dropout_7/Identity:output:0,autoencoder/model_6/flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????-?
1autoencoder/model_6/dense_6/MatMul/ReadVariableOpReadVariableOp:autoencoder_model_6_dense_6_matmul_readvariableop_resource*
_output_shapes
:	?-<*
dtype0?
"autoencoder/model_6/dense_6/MatMulMatMul.autoencoder/model_6/flatten_3/Reshape:output:09autoencoder/model_6/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<?
2autoencoder/model_6/dense_6/BiasAdd/ReadVariableOpReadVariableOp;autoencoder_model_6_dense_6_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0?
#autoencoder/model_6/dense_6/BiasAddBiasAdd,autoencoder/model_6/dense_6/MatMul:product:0:autoencoder/model_6/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<?
1autoencoder/model_7/dense_7/MatMul/ReadVariableOpReadVariableOp:autoencoder_model_7_dense_7_matmul_readvariableop_resource*
_output_shapes
:	<?-*
dtype0?
"autoencoder/model_7/dense_7/MatMulMatMul,autoencoder/model_6/dense_6/BiasAdd:output:09autoencoder/model_7/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????-?
2autoencoder/model_7/dense_7/BiasAdd/ReadVariableOpReadVariableOp;autoencoder_model_7_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?-*
dtype0?
#autoencoder/model_7/dense_7/BiasAddBiasAdd,autoencoder/model_7/dense_7/MatMul:product:0:autoencoder/model_7/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????-
#autoencoder/model_7/reshape_3/ShapeShape,autoencoder/model_7/dense_7/BiasAdd:output:0*
T0*
_output_shapes
:{
1autoencoder/model_7/reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3autoencoder/model_7/reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3autoencoder/model_7/reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+autoencoder/model_7/reshape_3/strided_sliceStridedSlice,autoencoder/model_7/reshape_3/Shape:output:0:autoencoder/model_7/reshape_3/strided_slice/stack:output:0<autoencoder/model_7/reshape_3/strided_slice/stack_1:output:0<autoencoder/model_7/reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
-autoencoder/model_7/reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :o
-autoencoder/model_7/reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-autoencoder/model_7/reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :(?
+autoencoder/model_7/reshape_3/Reshape/shapePack4autoencoder/model_7/reshape_3/strided_slice:output:06autoencoder/model_7/reshape_3/Reshape/shape/1:output:06autoencoder/model_7/reshape_3/Reshape/shape/2:output:06autoencoder/model_7/reshape_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
%autoencoder/model_7/reshape_3/ReshapeReshape,autoencoder/model_7/dense_7/BiasAdd:output:04autoencoder/model_7/reshape_3/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????(?
,autoencoder/model_7/conv2d_transpose_6/ShapeShape.autoencoder/model_7/reshape_3/Reshape:output:0*
T0*
_output_shapes
:?
:autoencoder/model_7/conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
<autoencoder/model_7/conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
<autoencoder/model_7/conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
4autoencoder/model_7/conv2d_transpose_6/strided_sliceStridedSlice5autoencoder/model_7/conv2d_transpose_6/Shape:output:0Cautoencoder/model_7/conv2d_transpose_6/strided_slice/stack:output:0Eautoencoder/model_7/conv2d_transpose_6/strided_slice/stack_1:output:0Eautoencoder/model_7/conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.autoencoder/model_7/conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :p
.autoencoder/model_7/conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :p
.autoencoder/model_7/conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :(?
,autoencoder/model_7/conv2d_transpose_6/stackPack=autoencoder/model_7/conv2d_transpose_6/strided_slice:output:07autoencoder/model_7/conv2d_transpose_6/stack/1:output:07autoencoder/model_7/conv2d_transpose_6/stack/2:output:07autoencoder/model_7/conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:?
<autoencoder/model_7/conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
>autoencoder/model_7/conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
>autoencoder/model_7/conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
6autoencoder/model_7/conv2d_transpose_6/strided_slice_1StridedSlice5autoencoder/model_7/conv2d_transpose_6/stack:output:0Eautoencoder/model_7/conv2d_transpose_6/strided_slice_1/stack:output:0Gautoencoder/model_7/conv2d_transpose_6/strided_slice_1/stack_1:output:0Gautoencoder/model_7/conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Fautoencoder/model_7/conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOpOautoencoder_model_7_conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:((*
dtype0?
7autoencoder/model_7/conv2d_transpose_6/conv2d_transposeConv2DBackpropInput5autoencoder/model_7/conv2d_transpose_6/stack:output:0Nautoencoder/model_7/conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0.autoencoder/model_7/reshape_3/Reshape:output:0*
T0*/
_output_shapes
:?????????(*
paddingSAME*
strides
?
=autoencoder/model_7/conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOpFautoencoder_model_7_conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
.autoencoder/model_7/conv2d_transpose_6/BiasAddBiasAdd@autoencoder/model_7/conv2d_transpose_6/conv2d_transpose:output:0Eautoencoder/model_7/conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(?
+autoencoder/model_7/conv2d_transpose_6/TanhTanh7autoencoder/model_7/conv2d_transpose_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(?
3autoencoder/model_7/conv2d_25/Conv2D/ReadVariableOpReadVariableOp<autoencoder_model_7_conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:((*
dtype0?
$autoencoder/model_7/conv2d_25/Conv2DConv2D/autoencoder/model_7/conv2d_transpose_6/Tanh:y:0;autoencoder/model_7/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
paddingSAME*
strides
?
4autoencoder/model_7/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp=autoencoder_model_7_conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
%autoencoder/model_7/conv2d_25/BiasAddBiasAdd-autoencoder/model_7/conv2d_25/Conv2D:output:0<autoencoder/model_7/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(?
"autoencoder/model_7/conv2d_25/TanhTanh.autoencoder/model_7/conv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(?
,autoencoder/model_7/conv2d_transpose_7/ShapeShape&autoencoder/model_7/conv2d_25/Tanh:y:0*
T0*
_output_shapes
:?
:autoencoder/model_7/conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
<autoencoder/model_7/conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
<autoencoder/model_7/conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
4autoencoder/model_7/conv2d_transpose_7/strided_sliceStridedSlice5autoencoder/model_7/conv2d_transpose_7/Shape:output:0Cautoencoder/model_7/conv2d_transpose_7/strided_slice/stack:output:0Eautoencoder/model_7/conv2d_transpose_7/strided_slice/stack_1:output:0Eautoencoder/model_7/conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.autoencoder/model_7/conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :0p
.autoencoder/model_7/conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :0p
.autoencoder/model_7/conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
,autoencoder/model_7/conv2d_transpose_7/stackPack=autoencoder/model_7/conv2d_transpose_7/strided_slice:output:07autoencoder/model_7/conv2d_transpose_7/stack/1:output:07autoencoder/model_7/conv2d_transpose_7/stack/2:output:07autoencoder/model_7/conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:?
<autoencoder/model_7/conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
>autoencoder/model_7/conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
>autoencoder/model_7/conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
6autoencoder/model_7/conv2d_transpose_7/strided_slice_1StridedSlice5autoencoder/model_7/conv2d_transpose_7/stack:output:0Eautoencoder/model_7/conv2d_transpose_7/strided_slice_1/stack:output:0Gautoencoder/model_7/conv2d_transpose_7/strided_slice_1/stack_1:output:0Gautoencoder/model_7/conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Fautoencoder/model_7/conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOpOautoencoder_model_7_conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
:(*
dtype0?
7autoencoder/model_7/conv2d_transpose_7/conv2d_transposeConv2DBackpropInput5autoencoder/model_7/conv2d_transpose_7/stack:output:0Nautoencoder/model_7/conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0&autoencoder/model_7/conv2d_25/Tanh:y:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
=autoencoder/model_7/conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOpFautoencoder_model_7_conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
.autoencoder/model_7/conv2d_transpose_7/BiasAddBiasAdd@autoencoder/model_7/conv2d_transpose_7/conv2d_transpose:output:0Eautoencoder/model_7/conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
+autoencoder/model_7/conv2d_transpose_7/TanhTanh7autoencoder/model_7/conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00?
3autoencoder/model_7/conv2d_26/Conv2D/ReadVariableOpReadVariableOp<autoencoder_model_7_conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
$autoencoder/model_7/conv2d_26/Conv2DConv2D/autoencoder/model_7/conv2d_transpose_7/Tanh:y:0;autoencoder/model_7/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
4autoencoder/model_7/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp=autoencoder_model_7_conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
%autoencoder/model_7/conv2d_26/BiasAddBiasAdd-autoencoder/model_7/conv2d_26/Conv2D:output:0<autoencoder/model_7/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
"autoencoder/model_7/conv2d_26/TanhTanh.autoencoder/model_7/conv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00?
3autoencoder/model_7/conv2d_27/Conv2D/ReadVariableOpReadVariableOp<autoencoder_model_7_conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
$autoencoder/model_7/conv2d_27/Conv2DConv2D&autoencoder/model_7/conv2d_26/Tanh:y:0;autoencoder/model_7/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
4autoencoder/model_7/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp=autoencoder_model_7_conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
%autoencoder/model_7/conv2d_27/BiasAddBiasAdd-autoencoder/model_7/conv2d_27/Conv2D:output:0<autoencoder/model_7/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
%autoencoder/model_7/conv2d_27/SigmoidSigmoid.autoencoder/model_7/conv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00?
IdentityIdentity)autoencoder/model_7/conv2d_27/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????00?

NoOpNoOp5^autoencoder/model_6/conv2d_21/BiasAdd/ReadVariableOp4^autoencoder/model_6/conv2d_21/Conv2D/ReadVariableOp5^autoencoder/model_6/conv2d_22/BiasAdd/ReadVariableOp4^autoencoder/model_6/conv2d_22/Conv2D/ReadVariableOp5^autoencoder/model_6/conv2d_23/BiasAdd/ReadVariableOp4^autoencoder/model_6/conv2d_23/Conv2D/ReadVariableOp5^autoencoder/model_6/conv2d_24/BiasAdd/ReadVariableOp4^autoencoder/model_6/conv2d_24/Conv2D/ReadVariableOp3^autoencoder/model_6/dense_6/BiasAdd/ReadVariableOp2^autoencoder/model_6/dense_6/MatMul/ReadVariableOp5^autoencoder/model_7/conv2d_25/BiasAdd/ReadVariableOp4^autoencoder/model_7/conv2d_25/Conv2D/ReadVariableOp5^autoencoder/model_7/conv2d_26/BiasAdd/ReadVariableOp4^autoencoder/model_7/conv2d_26/Conv2D/ReadVariableOp5^autoencoder/model_7/conv2d_27/BiasAdd/ReadVariableOp4^autoencoder/model_7/conv2d_27/Conv2D/ReadVariableOp>^autoencoder/model_7/conv2d_transpose_6/BiasAdd/ReadVariableOpG^autoencoder/model_7/conv2d_transpose_6/conv2d_transpose/ReadVariableOp>^autoencoder/model_7/conv2d_transpose_7/BiasAdd/ReadVariableOpG^autoencoder/model_7/conv2d_transpose_7/conv2d_transpose/ReadVariableOp3^autoencoder/model_7/dense_7/BiasAdd/ReadVariableOp2^autoencoder/model_7/dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????00: : : : : : : : : : : : : : : : : : : : : : 2l
4autoencoder/model_6/conv2d_21/BiasAdd/ReadVariableOp4autoencoder/model_6/conv2d_21/BiasAdd/ReadVariableOp2j
3autoencoder/model_6/conv2d_21/Conv2D/ReadVariableOp3autoencoder/model_6/conv2d_21/Conv2D/ReadVariableOp2l
4autoencoder/model_6/conv2d_22/BiasAdd/ReadVariableOp4autoencoder/model_6/conv2d_22/BiasAdd/ReadVariableOp2j
3autoencoder/model_6/conv2d_22/Conv2D/ReadVariableOp3autoencoder/model_6/conv2d_22/Conv2D/ReadVariableOp2l
4autoencoder/model_6/conv2d_23/BiasAdd/ReadVariableOp4autoencoder/model_6/conv2d_23/BiasAdd/ReadVariableOp2j
3autoencoder/model_6/conv2d_23/Conv2D/ReadVariableOp3autoencoder/model_6/conv2d_23/Conv2D/ReadVariableOp2l
4autoencoder/model_6/conv2d_24/BiasAdd/ReadVariableOp4autoencoder/model_6/conv2d_24/BiasAdd/ReadVariableOp2j
3autoencoder/model_6/conv2d_24/Conv2D/ReadVariableOp3autoencoder/model_6/conv2d_24/Conv2D/ReadVariableOp2h
2autoencoder/model_6/dense_6/BiasAdd/ReadVariableOp2autoencoder/model_6/dense_6/BiasAdd/ReadVariableOp2f
1autoencoder/model_6/dense_6/MatMul/ReadVariableOp1autoencoder/model_6/dense_6/MatMul/ReadVariableOp2l
4autoencoder/model_7/conv2d_25/BiasAdd/ReadVariableOp4autoencoder/model_7/conv2d_25/BiasAdd/ReadVariableOp2j
3autoencoder/model_7/conv2d_25/Conv2D/ReadVariableOp3autoencoder/model_7/conv2d_25/Conv2D/ReadVariableOp2l
4autoencoder/model_7/conv2d_26/BiasAdd/ReadVariableOp4autoencoder/model_7/conv2d_26/BiasAdd/ReadVariableOp2j
3autoencoder/model_7/conv2d_26/Conv2D/ReadVariableOp3autoencoder/model_7/conv2d_26/Conv2D/ReadVariableOp2l
4autoencoder/model_7/conv2d_27/BiasAdd/ReadVariableOp4autoencoder/model_7/conv2d_27/BiasAdd/ReadVariableOp2j
3autoencoder/model_7/conv2d_27/Conv2D/ReadVariableOp3autoencoder/model_7/conv2d_27/Conv2D/ReadVariableOp2~
=autoencoder/model_7/conv2d_transpose_6/BiasAdd/ReadVariableOp=autoencoder/model_7/conv2d_transpose_6/BiasAdd/ReadVariableOp2?
Fautoencoder/model_7/conv2d_transpose_6/conv2d_transpose/ReadVariableOpFautoencoder/model_7/conv2d_transpose_6/conv2d_transpose/ReadVariableOp2~
=autoencoder/model_7/conv2d_transpose_7/BiasAdd/ReadVariableOp=autoencoder/model_7/conv2d_transpose_7/BiasAdd/ReadVariableOp2?
Fautoencoder/model_7/conv2d_transpose_7/conv2d_transpose/ReadVariableOpFautoencoder/model_7/conv2d_transpose_7/conv2d_transpose/ReadVariableOp2h
2autoencoder/model_7/dense_7/BiasAdd/ReadVariableOp2autoencoder/model_7/dense_7/BiasAdd/ReadVariableOp2f
1autoencoder/model_7/dense_7/MatMul/ReadVariableOp1autoencoder/model_7/dense_7/MatMul/ReadVariableOp:^ Z
/
_output_shapes
:?????????00
'
_user_specified_nameencoder_input
?
c
*__inference_dropout_7_layer_call_fn_382545

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_380562w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?

?
E__inference_conv2d_26_layer_call_and_return_conditional_losses_380966

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00X
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????00_
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:?????????00w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????00: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
,__inference_autoencoder_layer_call_fn_381708

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:(
	unknown_4:(#
	unknown_5:((
	unknown_6:(
	unknown_7:	?-<
	unknown_8:<
	unknown_9:	<?-

unknown_10:	?-$

unknown_11:((

unknown_12:($

unknown_13:((

unknown_14:($

unknown_15:(

unknown_16:$

unknown_17:

unknown_18:$

unknown_19:

unknown_20:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_autoencoder_layer_call_and_return_conditional_losses_381309w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????00: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
G__inference_autoencoder_layer_call_and_return_conditional_losses_381457

inputs(
model_6_381410:
model_6_381412:(
model_6_381414:
model_6_381416:(
model_6_381418:(
model_6_381420:((
model_6_381422:((
model_6_381424:(!
model_6_381426:	?-<
model_6_381428:<!
model_7_381431:	<?-
model_7_381433:	?-(
model_7_381435:((
model_7_381437:((
model_7_381439:((
model_7_381441:((
model_7_381443:(
model_7_381445:(
model_7_381447:
model_7_381449:(
model_7_381451:
model_7_381453:
identity??model_6/StatefulPartitionedCall?model_7/StatefulPartitionedCall?
model_6/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_6_381410model_6_381412model_6_381414model_6_381416model_6_381418model_6_381420model_6_381422model_6_381424model_6_381426model_6_381428*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_6_layer_call_and_return_conditional_losses_380687?
model_7/StatefulPartitionedCallStatefulPartitionedCall(model_6/StatefulPartitionedCall:output:0model_7_381431model_7_381433model_7_381435model_7_381437model_7_381439model_7_381441model_7_381443model_7_381445model_7_381447model_7_381449model_7_381451model_7_381453*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_7_layer_call_and_return_conditional_losses_381129
IdentityIdentity(model_7/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp ^model_6/StatefulPartitionedCall ^model_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????00: : : : : : : : : : : : : : : : : : : : : : 2B
model_6/StatefulPartitionedCallmodel_6/StatefulPartitionedCall2B
model_7/StatefulPartitionedCallmodel_7/StatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
(__inference_dense_6_layer_call_fn_382582

inputs
unknown:	?-<
	unknown_0:<
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_380496o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????-: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????-
 
_user_specified_nameinputs
?

?
(__inference_model_6_layer_call_fn_380735
encoder_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:(
	unknown_4:(#
	unknown_5:((
	unknown_6:(
	unknown_7:	?-<
	unknown_8:<
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_6_layer_call_and_return_conditional_losses_380687o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????00: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:?????????00
'
_user_specified_nameencoder_input
?

d
E__inference_dropout_6_layer_call_and_return_conditional_losses_380605

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_reshape_3_layer_call_and_return_conditional_losses_380926

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :(?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????(`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????-:P L
(
_output_shapes
:??????????-
 
_user_specified_nameinputs
?
c
E__inference_dropout_6_layer_call_and_return_conditional_losses_380435

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_dense_7_layer_call_fn_382601

inputs
unknown:	<?-
	unknown_0:	?-
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_380906p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????-`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????<: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?!
?
N__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_380882

inputsB
(conv2d_transpose_readvariableop_resource:(-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:(*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????j
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????q
IdentityIdentityTanh:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????(
 
_user_specified_nameinputs
?
F
*__inference_reshape_3_layer_call_fn_382616

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_reshape_3_layer_call_and_return_conditional_losses_380926h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????-:P L
(
_output_shapes
:??????????-
 
_user_specified_nameinputs
?
?
*__inference_conv2d_22_layer_call_fn_382457

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_22_layer_call_and_return_conditional_losses_380424w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????00: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?

?
E__inference_conv2d_26_layer_call_and_return_conditional_losses_382756

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00X
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????00_
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:?????????00w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????00: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?e
?

C__inference_model_7_layer_call_and_return_conditional_losses_382428

inputs9
&dense_7_matmul_readvariableop_resource:	<?-6
'dense_7_biasadd_readvariableop_resource:	?-U
;conv2d_transpose_6_conv2d_transpose_readvariableop_resource:((@
2conv2d_transpose_6_biasadd_readvariableop_resource:(B
(conv2d_25_conv2d_readvariableop_resource:((7
)conv2d_25_biasadd_readvariableop_resource:(U
;conv2d_transpose_7_conv2d_transpose_readvariableop_resource:(@
2conv2d_transpose_7_biasadd_readvariableop_resource:B
(conv2d_26_conv2d_readvariableop_resource:7
)conv2d_26_biasadd_readvariableop_resource:B
(conv2d_27_conv2d_readvariableop_resource:7
)conv2d_27_biasadd_readvariableop_resource:
identity?? conv2d_25/BiasAdd/ReadVariableOp?conv2d_25/Conv2D/ReadVariableOp? conv2d_26/BiasAdd/ReadVariableOp?conv2d_26/Conv2D/ReadVariableOp? conv2d_27/BiasAdd/ReadVariableOp?conv2d_27/Conv2D/ReadVariableOp?)conv2d_transpose_6/BiasAdd/ReadVariableOp?2conv2d_transpose_6/conv2d_transpose/ReadVariableOp?)conv2d_transpose_7/BiasAdd/ReadVariableOp?2conv2d_transpose_7/conv2d_transpose/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	<?-*
dtype0z
dense_7/MatMulMatMulinputs%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????-?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?-*
dtype0?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????-W
reshape_3/ShapeShapedense_7/BiasAdd:output:0*
T0*
_output_shapes
:g
reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape_3/strided_sliceStridedSlicereshape_3/Shape:output:0&reshape_3/strided_slice/stack:output:0(reshape_3/strided_slice/stack_1:output:0(reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :(?
reshape_3/Reshape/shapePack reshape_3/strided_slice:output:0"reshape_3/Reshape/shape/1:output:0"reshape_3/Reshape/shape/2:output:0"reshape_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
reshape_3/ReshapeReshapedense_7/BiasAdd:output:0 reshape_3/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????(b
conv2d_transpose_6/ShapeShapereshape_3/Reshape:output:0*
T0*
_output_shapes
:p
&conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_6/strided_sliceStridedSlice!conv2d_transpose_6/Shape:output:0/conv2d_transpose_6/strided_slice/stack:output:01conv2d_transpose_6/strided_slice/stack_1:output:01conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :(?
conv2d_transpose_6/stackPack)conv2d_transpose_6/strided_slice:output:0#conv2d_transpose_6/stack/1:output:0#conv2d_transpose_6/stack/2:output:0#conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_6/strided_slice_1StridedSlice!conv2d_transpose_6/stack:output:01conv2d_transpose_6/strided_slice_1/stack:output:03conv2d_transpose_6/strided_slice_1/stack_1:output:03conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:((*
dtype0?
#conv2d_transpose_6/conv2d_transposeConv2DBackpropInput!conv2d_transpose_6/stack:output:0:conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0reshape_3/Reshape:output:0*
T0*/
_output_shapes
:?????????(*
paddingSAME*
strides
?
)conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
conv2d_transpose_6/BiasAddBiasAdd,conv2d_transpose_6/conv2d_transpose:output:01conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(~
conv2d_transpose_6/TanhTanh#conv2d_transpose_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(?
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:((*
dtype0?
conv2d_25/Conv2DConv2Dconv2d_transpose_6/Tanh:y:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
paddingSAME*
strides
?
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(l
conv2d_25/TanhTanhconv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(Z
conv2d_transpose_7/ShapeShapeconv2d_25/Tanh:y:0*
T0*
_output_shapes
:p
&conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_7/strided_sliceStridedSlice!conv2d_transpose_7/Shape:output:0/conv2d_transpose_7/strided_slice/stack:output:01conv2d_transpose_7/strided_slice/stack_1:output:01conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :0\
conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :0\
conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_7/stackPack)conv2d_transpose_7/strided_slice:output:0#conv2d_transpose_7/stack/1:output:0#conv2d_transpose_7/stack/2:output:0#conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_7/strided_slice_1StridedSlice!conv2d_transpose_7/stack:output:01conv2d_transpose_7/strided_slice_1/stack:output:03conv2d_transpose_7/strided_slice_1/stack_1:output:03conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
:(*
dtype0?
#conv2d_transpose_7/conv2d_transposeConv2DBackpropInput!conv2d_transpose_7/stack:output:0:conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0conv2d_25/Tanh:y:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
)conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_7/BiasAddBiasAdd,conv2d_transpose_7/conv2d_transpose:output:01conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00~
conv2d_transpose_7/TanhTanh#conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00?
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_26/Conv2DConv2Dconv2d_transpose_7/Tanh:y:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00l
conv2d_26/TanhTanhconv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00?
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_27/Conv2DConv2Dconv2d_26/Tanh:y:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00r
conv2d_27/SigmoidSigmoidconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00l
IdentityIdentityconv2d_27/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp*^conv2d_transpose_6/BiasAdd/ReadVariableOp3^conv2d_transpose_6/conv2d_transpose/ReadVariableOp*^conv2d_transpose_7/BiasAdd/ReadVariableOp3^conv2d_transpose_7/conv2d_transpose/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????<: : : : : : : : : : : : 2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2V
)conv2d_transpose_6/BiasAdd/ReadVariableOp)conv2d_transpose_6/BiasAdd/ReadVariableOp2h
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_7/BiasAdd/ReadVariableOp)conv2d_transpose_7/BiasAdd/ReadVariableOp2h
2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?
a
E__inference_reshape_3_layer_call_and_return_conditional_losses_382630

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :(?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????(`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????-:P L
(
_output_shapes
:??????????-
 
_user_specified_nameinputs
?

?
E__inference_conv2d_25_layer_call_and_return_conditional_losses_380944

inputs8
conv2d_readvariableop_resource:((-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:((*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(X
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????(_
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:?????????(w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?

?
(__inference_model_6_layer_call_fn_382085

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:(
	unknown_4:(#
	unknown_5:((
	unknown_6:(
	unknown_7:	?-<
	unknown_8:<
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_6_layer_call_and_return_conditional_losses_380503o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????00: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?!
?
N__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_382673

inputsB
(conv2d_transpose_readvariableop_resource:((-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :(y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:((*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????(*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????(j
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????(q
IdentityIdentityTanh:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????(?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????(
 
_user_specified_nameinputs
?
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_380484

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????-Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????-"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
G__inference_autoencoder_layer_call_and_return_conditional_losses_381603
encoder_input(
model_6_381556:
model_6_381558:(
model_6_381560:
model_6_381562:(
model_6_381564:(
model_6_381566:((
model_6_381568:((
model_6_381570:(!
model_6_381572:	?-<
model_6_381574:<!
model_7_381577:	<?-
model_7_381579:	?-(
model_7_381581:((
model_7_381583:((
model_7_381585:((
model_7_381587:((
model_7_381589:(
model_7_381591:(
model_7_381593:
model_7_381595:(
model_7_381597:
model_7_381599:
identity??model_6/StatefulPartitionedCall?model_7/StatefulPartitionedCall?
model_6/StatefulPartitionedCallStatefulPartitionedCallencoder_inputmodel_6_381556model_6_381558model_6_381560model_6_381562model_6_381564model_6_381566model_6_381568model_6_381570model_6_381572model_6_381574*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_6_layer_call_and_return_conditional_losses_380503?
model_7/StatefulPartitionedCallStatefulPartitionedCall(model_6/StatefulPartitionedCall:output:0model_7_381577model_7_381579model_7_381581model_7_381583model_7_381585model_7_381587model_7_381589model_7_381591model_7_381593model_7_381595model_7_381597model_7_381599*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_7_layer_call_and_return_conditional_losses_380990
IdentityIdentity(model_7/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp ^model_6/StatefulPartitionedCall ^model_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????00: : : : : : : : : : : : : : : : : : : : : : 2B
model_6/StatefulPartitionedCallmodel_6/StatefulPartitionedCall2B
model_7/StatefulPartitionedCallmodel_7/StatefulPartitionedCall:^ Z
/
_output_shapes
:?????????00
'
_user_specified_nameencoder_input
?%
?
C__inference_model_7_layer_call_and_return_conditional_losses_381129

inputs!
dense_7_381097:	<?-
dense_7_381099:	?-3
conv2d_transpose_6_381103:(('
conv2d_transpose_6_381105:(*
conv2d_25_381108:((
conv2d_25_381110:(3
conv2d_transpose_7_381113:('
conv2d_transpose_7_381115:*
conv2d_26_381118:
conv2d_26_381120:*
conv2d_27_381123:
conv2d_27_381125:
identity??!conv2d_25/StatefulPartitionedCall?!conv2d_26/StatefulPartitionedCall?!conv2d_27/StatefulPartitionedCall?*conv2d_transpose_6/StatefulPartitionedCall?*conv2d_transpose_7/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCallinputsdense_7_381097dense_7_381099*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_380906?
reshape_3/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_reshape_3_layer_call_and_return_conditional_losses_380926?
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall"reshape_3/PartitionedCall:output:0conv2d_transpose_6_381103conv2d_transpose_6_381105*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_380837?
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0conv2d_25_381108conv2d_25_381110*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_25_layer_call_and_return_conditional_losses_380944?
*conv2d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0conv2d_transpose_7_381113conv2d_transpose_7_381115*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_380882?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_7/StatefulPartitionedCall:output:0conv2d_26_381118conv2d_26_381120*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_26_layer_call_and_return_conditional_losses_380966?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0conv2d_27_381123conv2d_27_381125*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_27_layer_call_and_return_conditional_losses_380983?
IdentityIdentity*conv2d_27/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall+^conv2d_transpose_6/StatefulPartitionedCall+^conv2d_transpose_7/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????<: : : : : : : : : : : : 2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2X
*conv2d_transpose_7/StatefulPartitionedCall*conv2d_transpose_7/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?
?
,__inference_autoencoder_layer_call_fn_381356
encoder_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:(
	unknown_4:(#
	unknown_5:((
	unknown_6:(
	unknown_7:	?-<
	unknown_8:<
	unknown_9:	<?-

unknown_10:	?-$

unknown_11:((

unknown_12:($

unknown_13:((

unknown_14:($

unknown_15:(

unknown_16:$

unknown_17:

unknown_18:$

unknown_19:

unknown_20:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_autoencoder_layer_call_and_return_conditional_losses_381309w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????00: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:?????????00
'
_user_specified_nameencoder_input
?
?
(__inference_model_7_layer_call_fn_381185
input_4
unknown:	<?-
	unknown_0:	?-#
	unknown_1:((
	unknown_2:(#
	unknown_3:((
	unknown_4:(#
	unknown_5:(
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_7_layer_call_and_return_conditional_losses_381129w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????<: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????<
!
_user_specified_name	input_4
?

?
E__inference_conv2d_24_layer_call_and_return_conditional_losses_380465

inputs8
conv2d_readvariableop_resource:((-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:((*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(X
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????(_
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:?????????(w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?

?
E__inference_conv2d_22_layer_call_and_return_conditional_losses_380424

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????X
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????_
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????00: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
*__inference_conv2d_24_layer_call_fn_382524

inputs!
unknown:((
	unknown_0:(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_24_layer_call_and_return_conditional_losses_380465w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?%
?
C__inference_model_7_layer_call_and_return_conditional_losses_381220
input_4!
dense_7_381188:	<?-
dense_7_381190:	?-3
conv2d_transpose_6_381194:(('
conv2d_transpose_6_381196:(*
conv2d_25_381199:((
conv2d_25_381201:(3
conv2d_transpose_7_381204:('
conv2d_transpose_7_381206:*
conv2d_26_381209:
conv2d_26_381211:*
conv2d_27_381214:
conv2d_27_381216:
identity??!conv2d_25/StatefulPartitionedCall?!conv2d_26/StatefulPartitionedCall?!conv2d_27/StatefulPartitionedCall?*conv2d_transpose_6/StatefulPartitionedCall?*conv2d_transpose_7/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_7_381188dense_7_381190*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_380906?
reshape_3/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_reshape_3_layer_call_and_return_conditional_losses_380926?
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall"reshape_3/PartitionedCall:output:0conv2d_transpose_6_381194conv2d_transpose_6_381196*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_380837?
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0conv2d_25_381199conv2d_25_381201*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_25_layer_call_and_return_conditional_losses_380944?
*conv2d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0conv2d_transpose_7_381204conv2d_transpose_7_381206*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_380882?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_7/StatefulPartitionedCall:output:0conv2d_26_381209conv2d_26_381211*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_26_layer_call_and_return_conditional_losses_380966?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0conv2d_27_381214conv2d_27_381216*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_27_layer_call_and_return_conditional_losses_380983?
IdentityIdentity*conv2d_27/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall+^conv2d_transpose_6/StatefulPartitionedCall+^conv2d_transpose_7/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????<: : : : : : : : : : : : 2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2X
*conv2d_transpose_7/StatefulPartitionedCall*conv2d_transpose_7/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:P L
'
_output_shapes
:?????????<
!
_user_specified_name	input_4
?

?
E__inference_conv2d_24_layer_call_and_return_conditional_losses_382535

inputs8
conv2d_readvariableop_resource:((-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:((*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(X
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????(_
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:?????????(w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?

?
E__inference_conv2d_25_layer_call_and_return_conditional_losses_382693

inputs8
conv2d_readvariableop_resource:((-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:((*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(X
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????(_
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:?????????(w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
(__inference_model_7_layer_call_fn_382266

inputs
unknown:	<?-
	unknown_0:	?-#
	unknown_1:((
	unknown_2:(#
	unknown_3:((
	unknown_4:(#
	unknown_5:(
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_7_layer_call_and_return_conditional_losses_381129w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????<: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?

?
E__inference_conv2d_23_layer_call_and_return_conditional_losses_382515

inputs8
conv2d_readvariableop_resource:(-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:(*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(X
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????(_
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:?????????(w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
3__inference_conv2d_transpose_6_layer_call_fn_382639

inputs!
unknown:((
	unknown_0:(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_380837?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????(: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????(
 
_user_specified_nameinputs
?%
?
C__inference_model_6_layer_call_and_return_conditional_losses_380767
encoder_input*
conv2d_21_380738:
conv2d_21_380740:*
conv2d_22_380743:
conv2d_22_380745:*
conv2d_23_380749:(
conv2d_23_380751:(*
conv2d_24_380754:((
conv2d_24_380756:(!
dense_6_380761:	?-<
dense_6_380763:<
identity??!conv2d_21/StatefulPartitionedCall?!conv2d_22/StatefulPartitionedCall?!conv2d_23/StatefulPartitionedCall?!conv2d_24/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCallencoder_inputconv2d_21_380738conv2d_21_380740*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_21_layer_call_and_return_conditional_losses_380407?
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0conv2d_22_380743conv2d_22_380745*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_22_layer_call_and_return_conditional_losses_380424?
dropout_6/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_380435?
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall"dropout_6/PartitionedCall:output:0conv2d_23_380749conv2d_23_380751*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_23_layer_call_and_return_conditional_losses_380448?
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0conv2d_24_380754conv2d_24_380756*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_24_layer_call_and_return_conditional_losses_380465?
dropout_7/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_380476?
flatten_3/PartitionedCallPartitionedCall"dropout_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????-* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_380484?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_380761dense_6_380763*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_380496w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????<?
NoOpNoOp"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????00: : : : : : : : : : 2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:^ Z
/
_output_shapes
:?????????00
'
_user_specified_nameencoder_input
?
c
E__inference_dropout_6_layer_call_and_return_conditional_losses_382483

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
__inference__traced_save_383018
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop/
+savev2_conv2d_21_kernel_read_readvariableop-
)savev2_conv2d_21_bias_read_readvariableop/
+savev2_conv2d_22_kernel_read_readvariableop-
)savev2_conv2d_22_bias_read_readvariableop/
+savev2_conv2d_23_kernel_read_readvariableop-
)savev2_conv2d_23_bias_read_readvariableop/
+savev2_conv2d_24_kernel_read_readvariableop-
)savev2_conv2d_24_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop8
4savev2_conv2d_transpose_6_kernel_read_readvariableop6
2savev2_conv2d_transpose_6_bias_read_readvariableop/
+savev2_conv2d_25_kernel_read_readvariableop-
)savev2_conv2d_25_bias_read_readvariableop8
4savev2_conv2d_transpose_7_kernel_read_readvariableop6
2savev2_conv2d_transpose_7_bias_read_readvariableop/
+savev2_conv2d_26_kernel_read_readvariableop-
)savev2_conv2d_26_bias_read_readvariableop/
+savev2_conv2d_27_kernel_read_readvariableop-
)savev2_conv2d_27_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_conv2d_21_kernel_m_read_readvariableop4
0savev2_adam_conv2d_21_bias_m_read_readvariableop6
2savev2_adam_conv2d_22_kernel_m_read_readvariableop4
0savev2_adam_conv2d_22_bias_m_read_readvariableop6
2savev2_adam_conv2d_23_kernel_m_read_readvariableop4
0savev2_adam_conv2d_23_bias_m_read_readvariableop6
2savev2_adam_conv2d_24_kernel_m_read_readvariableop4
0savev2_adam_conv2d_24_bias_m_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableop4
0savev2_adam_dense_7_kernel_m_read_readvariableop2
.savev2_adam_dense_7_bias_m_read_readvariableop?
;savev2_adam_conv2d_transpose_6_kernel_m_read_readvariableop=
9savev2_adam_conv2d_transpose_6_bias_m_read_readvariableop6
2savev2_adam_conv2d_25_kernel_m_read_readvariableop4
0savev2_adam_conv2d_25_bias_m_read_readvariableop?
;savev2_adam_conv2d_transpose_7_kernel_m_read_readvariableop=
9savev2_adam_conv2d_transpose_7_bias_m_read_readvariableop6
2savev2_adam_conv2d_26_kernel_m_read_readvariableop4
0savev2_adam_conv2d_26_bias_m_read_readvariableop6
2savev2_adam_conv2d_27_kernel_m_read_readvariableop4
0savev2_adam_conv2d_27_bias_m_read_readvariableop6
2savev2_adam_conv2d_21_kernel_v_read_readvariableop4
0savev2_adam_conv2d_21_bias_v_read_readvariableop6
2savev2_adam_conv2d_22_kernel_v_read_readvariableop4
0savev2_adam_conv2d_22_bias_v_read_readvariableop6
2savev2_adam_conv2d_23_kernel_v_read_readvariableop4
0savev2_adam_conv2d_23_bias_v_read_readvariableop6
2savev2_adam_conv2d_24_kernel_v_read_readvariableop4
0savev2_adam_conv2d_24_bias_v_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableop4
0savev2_adam_dense_7_kernel_v_read_readvariableop2
.savev2_adam_dense_7_bias_v_read_readvariableop?
;savev2_adam_conv2d_transpose_6_kernel_v_read_readvariableop=
9savev2_adam_conv2d_transpose_6_bias_v_read_readvariableop6
2savev2_adam_conv2d_25_kernel_v_read_readvariableop4
0savev2_adam_conv2d_25_bias_v_read_readvariableop?
;savev2_adam_conv2d_transpose_7_kernel_v_read_readvariableop=
9savev2_adam_conv2d_transpose_7_bias_v_read_readvariableop6
2savev2_adam_conv2d_26_kernel_v_read_readvariableop4
0savev2_adam_conv2d_26_bias_v_read_readvariableop6
2savev2_adam_conv2d_27_kernel_v_read_readvariableop4
0savev2_adam_conv2d_27_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?"
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*?!
value?!B?!JB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*?
value?B?JB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop+savev2_conv2d_21_kernel_read_readvariableop)savev2_conv2d_21_bias_read_readvariableop+savev2_conv2d_22_kernel_read_readvariableop)savev2_conv2d_22_bias_read_readvariableop+savev2_conv2d_23_kernel_read_readvariableop)savev2_conv2d_23_bias_read_readvariableop+savev2_conv2d_24_kernel_read_readvariableop)savev2_conv2d_24_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop4savev2_conv2d_transpose_6_kernel_read_readvariableop2savev2_conv2d_transpose_6_bias_read_readvariableop+savev2_conv2d_25_kernel_read_readvariableop)savev2_conv2d_25_bias_read_readvariableop4savev2_conv2d_transpose_7_kernel_read_readvariableop2savev2_conv2d_transpose_7_bias_read_readvariableop+savev2_conv2d_26_kernel_read_readvariableop)savev2_conv2d_26_bias_read_readvariableop+savev2_conv2d_27_kernel_read_readvariableop)savev2_conv2d_27_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_conv2d_21_kernel_m_read_readvariableop0savev2_adam_conv2d_21_bias_m_read_readvariableop2savev2_adam_conv2d_22_kernel_m_read_readvariableop0savev2_adam_conv2d_22_bias_m_read_readvariableop2savev2_adam_conv2d_23_kernel_m_read_readvariableop0savev2_adam_conv2d_23_bias_m_read_readvariableop2savev2_adam_conv2d_24_kernel_m_read_readvariableop0savev2_adam_conv2d_24_bias_m_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop0savev2_adam_dense_7_kernel_m_read_readvariableop.savev2_adam_dense_7_bias_m_read_readvariableop;savev2_adam_conv2d_transpose_6_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_6_bias_m_read_readvariableop2savev2_adam_conv2d_25_kernel_m_read_readvariableop0savev2_adam_conv2d_25_bias_m_read_readvariableop;savev2_adam_conv2d_transpose_7_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_7_bias_m_read_readvariableop2savev2_adam_conv2d_26_kernel_m_read_readvariableop0savev2_adam_conv2d_26_bias_m_read_readvariableop2savev2_adam_conv2d_27_kernel_m_read_readvariableop0savev2_adam_conv2d_27_bias_m_read_readvariableop2savev2_adam_conv2d_21_kernel_v_read_readvariableop0savev2_adam_conv2d_21_bias_v_read_readvariableop2savev2_adam_conv2d_22_kernel_v_read_readvariableop0savev2_adam_conv2d_22_bias_v_read_readvariableop2savev2_adam_conv2d_23_kernel_v_read_readvariableop0savev2_adam_conv2d_23_bias_v_read_readvariableop2savev2_adam_conv2d_24_kernel_v_read_readvariableop0savev2_adam_conv2d_24_bias_v_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableop0savev2_adam_dense_7_kernel_v_read_readvariableop.savev2_adam_dense_7_bias_v_read_readvariableop;savev2_adam_conv2d_transpose_6_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_6_bias_v_read_readvariableop2savev2_adam_conv2d_25_kernel_v_read_readvariableop0savev2_adam_conv2d_25_bias_v_read_readvariableop;savev2_adam_conv2d_transpose_7_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_7_bias_v_read_readvariableop2savev2_adam_conv2d_26_kernel_v_read_readvariableop0savev2_adam_conv2d_26_bias_v_read_readvariableop2savev2_adam_conv2d_27_kernel_v_read_readvariableop0savev2_adam_conv2d_27_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *X
dtypesN
L2J	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : :::::(:(:((:(:	?-<:<:	<?-:?-:((:(:((:(:(:::::: : :::::(:(:((:(:	?-<:<:	<?-:?-:((:(:((:(:(::::::::::(:(:((:(:	?-<:<:	<?-:?-:((:(:((:(:(:::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 	

_output_shapes
::,
(
&
_output_shapes
:(: 

_output_shapes
:(:,(
&
_output_shapes
:((: 

_output_shapes
:(:%!

_output_shapes
:	?-<: 

_output_shapes
:<:%!

_output_shapes
:	<?-:!

_output_shapes	
:?-:,(
&
_output_shapes
:((: 

_output_shapes
:(:,(
&
_output_shapes
:((: 

_output_shapes
:(:,(
&
_output_shapes
:(: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::, (
&
_output_shapes
:: !

_output_shapes
::,"(
&
_output_shapes
:(: #

_output_shapes
:(:,$(
&
_output_shapes
:((: %

_output_shapes
:(:%&!

_output_shapes
:	?-<: '

_output_shapes
:<:%(!

_output_shapes
:	<?-:!)

_output_shapes	
:?-:,*(
&
_output_shapes
:((: +

_output_shapes
:(:,,(
&
_output_shapes
:((: -

_output_shapes
:(:,.(
&
_output_shapes
:(: /

_output_shapes
::,0(
&
_output_shapes
:: 1

_output_shapes
::,2(
&
_output_shapes
:: 3

_output_shapes
::,4(
&
_output_shapes
:: 5

_output_shapes
::,6(
&
_output_shapes
:: 7

_output_shapes
::,8(
&
_output_shapes
:(: 9

_output_shapes
:(:,:(
&
_output_shapes
:((: ;

_output_shapes
:(:%<!

_output_shapes
:	?-<: =

_output_shapes
:<:%>!

_output_shapes
:	<?-:!?

_output_shapes	
:?-:,@(
&
_output_shapes
:((: A

_output_shapes
:(:,B(
&
_output_shapes
:((: C

_output_shapes
:(:,D(
&
_output_shapes
:(: E

_output_shapes
::,F(
&
_output_shapes
:: G

_output_shapes
::,H(
&
_output_shapes
:: I

_output_shapes
::J

_output_shapes
: 
?
c
E__inference_dropout_7_layer_call_and_return_conditional_losses_382550

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????(c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????("!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?	
?
C__inference_dense_6_layer_call_and_return_conditional_losses_380496

inputs1
matmul_readvariableop_resource:	?-<-
biasadd_readvariableop_resource:<
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?-<*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????<w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????-: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????-
 
_user_specified_nameinputs
?	
?
C__inference_dense_7_layer_call_and_return_conditional_losses_382611

inputs1
matmul_readvariableop_resource:	<?-.
biasadd_readvariableop_resource:	?-
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	<?-*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????-s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?-*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????-`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????-w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?'
?
C__inference_model_6_layer_call_and_return_conditional_losses_380799
encoder_input*
conv2d_21_380770:
conv2d_21_380772:*
conv2d_22_380775:
conv2d_22_380777:*
conv2d_23_380781:(
conv2d_23_380783:(*
conv2d_24_380786:((
conv2d_24_380788:(!
dense_6_380793:	?-<
dense_6_380795:<
identity??!conv2d_21/StatefulPartitionedCall?!conv2d_22/StatefulPartitionedCall?!conv2d_23/StatefulPartitionedCall?!conv2d_24/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?!dropout_6/StatefulPartitionedCall?!dropout_7/StatefulPartitionedCall?
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCallencoder_inputconv2d_21_380770conv2d_21_380772*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_21_layer_call_and_return_conditional_losses_380407?
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0conv2d_22_380775conv2d_22_380777*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_22_layer_call_and_return_conditional_losses_380424?
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_380605?
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall*dropout_6/StatefulPartitionedCall:output:0conv2d_23_380781conv2d_23_380783*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_23_layer_call_and_return_conditional_losses_380448?
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0conv2d_24_380786conv2d_24_380788*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_24_layer_call_and_return_conditional_losses_380465?
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_380562?
flatten_3/PartitionedCallPartitionedCall*dropout_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????-* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_380484?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_380793dense_6_380795*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_380496w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????<?
NoOpNoOp"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????00: : : : : : : : : : 2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall:^ Z
/
_output_shapes
:?????????00
'
_user_specified_nameencoder_input
?

?
E__inference_conv2d_21_layer_call_and_return_conditional_losses_380407

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00X
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????00_
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:?????????00w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????00: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_382573

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????-Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????-"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?2
?
C__inference_model_6_layer_call_and_return_conditional_losses_382152

inputsB
(conv2d_21_conv2d_readvariableop_resource:7
)conv2d_21_biasadd_readvariableop_resource:B
(conv2d_22_conv2d_readvariableop_resource:7
)conv2d_22_biasadd_readvariableop_resource:B
(conv2d_23_conv2d_readvariableop_resource:(7
)conv2d_23_biasadd_readvariableop_resource:(B
(conv2d_24_conv2d_readvariableop_resource:((7
)conv2d_24_biasadd_readvariableop_resource:(9
&dense_6_matmul_readvariableop_resource:	?-<5
'dense_6_biasadd_readvariableop_resource:<
identity?? conv2d_21/BiasAdd/ReadVariableOp?conv2d_21/Conv2D/ReadVariableOp? conv2d_22/BiasAdd/ReadVariableOp?conv2d_22/Conv2D/ReadVariableOp? conv2d_23/BiasAdd/ReadVariableOp?conv2d_23/Conv2D/ReadVariableOp? conv2d_24/BiasAdd/ReadVariableOp?conv2d_24/Conv2D/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_21/Conv2DConv2Dinputs'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00l
conv2d_21/TanhTanhconv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00?
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_22/Conv2DConv2Dconv2d_21/Tanh:y:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????l
conv2d_22/TanhTanhconv2d_22/BiasAdd:output:0*
T0*/
_output_shapes
:?????????l
dropout_6/IdentityIdentityconv2d_22/Tanh:y:0*
T0*/
_output_shapes
:??????????
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype0?
conv2d_23/Conv2DConv2Ddropout_6/Identity:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
paddingSAME*
strides
?
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(l
conv2d_23/TanhTanhconv2d_23/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(?
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
:((*
dtype0?
conv2d_24/Conv2DConv2Dconv2d_23/Tanh:y:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
paddingSAME*
strides
?
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(l
conv2d_24/TanhTanhconv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(l
dropout_7/IdentityIdentityconv2d_24/Tanh:y:0*
T0*/
_output_shapes
:?????????(`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
flatten_3/ReshapeReshapedropout_7/Identity:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????-?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	?-<*
dtype0?
dense_6/MatMulMatMulflatten_3/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<g
IdentityIdentitydense_6/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????<?
NoOpNoOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????00: : : : : : : : : : 2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
c
E__inference_dropout_7_layer_call_and_return_conditional_losses_380476

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????(c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????("!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?A
?
C__inference_model_6_layer_call_and_return_conditional_losses_382208

inputsB
(conv2d_21_conv2d_readvariableop_resource:7
)conv2d_21_biasadd_readvariableop_resource:B
(conv2d_22_conv2d_readvariableop_resource:7
)conv2d_22_biasadd_readvariableop_resource:B
(conv2d_23_conv2d_readvariableop_resource:(7
)conv2d_23_biasadd_readvariableop_resource:(B
(conv2d_24_conv2d_readvariableop_resource:((7
)conv2d_24_biasadd_readvariableop_resource:(9
&dense_6_matmul_readvariableop_resource:	?-<5
'dense_6_biasadd_readvariableop_resource:<
identity?? conv2d_21/BiasAdd/ReadVariableOp?conv2d_21/Conv2D/ReadVariableOp? conv2d_22/BiasAdd/ReadVariableOp?conv2d_22/Conv2D/ReadVariableOp? conv2d_23/BiasAdd/ReadVariableOp?conv2d_23/Conv2D/ReadVariableOp? conv2d_24/BiasAdd/ReadVariableOp?conv2d_24/Conv2D/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_21/Conv2DConv2Dinputs'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00l
conv2d_21/TanhTanhconv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00?
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_22/Conv2DConv2Dconv2d_21/Tanh:y:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????l
conv2d_22/TanhTanhconv2d_22/BiasAdd:output:0*
T0*/
_output_shapes
:?????????\
dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_6/dropout/MulMulconv2d_22/Tanh:y:0 dropout_6/dropout/Const:output:0*
T0*/
_output_shapes
:?????????Y
dropout_6/dropout/ShapeShapeconv2d_22/Tanh:y:0*
T0*
_output_shapes
:?
.dropout_6/dropout/random_uniform/RandomUniformRandomUniform dropout_6/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype0e
 dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_6/dropout/GreaterEqualGreaterEqual7dropout_6/dropout/random_uniform/RandomUniform:output:0)dropout_6/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:??????????
dropout_6/dropout/CastCast"dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:??????????
dropout_6/dropout/Mul_1Muldropout_6/dropout/Mul:z:0dropout_6/dropout/Cast:y:0*
T0*/
_output_shapes
:??????????
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype0?
conv2d_23/Conv2DConv2Ddropout_6/dropout/Mul_1:z:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
paddingSAME*
strides
?
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(l
conv2d_23/TanhTanhconv2d_23/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(?
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
:((*
dtype0?
conv2d_24/Conv2DConv2Dconv2d_23/Tanh:y:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
paddingSAME*
strides
?
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(l
conv2d_24/TanhTanhconv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(\
dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_7/dropout/MulMulconv2d_24/Tanh:y:0 dropout_7/dropout/Const:output:0*
T0*/
_output_shapes
:?????????(Y
dropout_7/dropout/ShapeShapeconv2d_24/Tanh:y:0*
T0*
_output_shapes
:?
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????(*
dtype0e
 dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_7/dropout/GreaterEqualGreaterEqual7dropout_7/dropout/random_uniform/RandomUniform:output:0)dropout_7/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????(?
dropout_7/dropout/CastCast"dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????(?
dropout_7/dropout/Mul_1Muldropout_7/dropout/Mul:z:0dropout_7/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????(`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
flatten_3/ReshapeReshapedropout_7/dropout/Mul_1:z:0flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????-?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	?-<*
dtype0?
dense_6/MatMulMatMulflatten_3/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<g
IdentityIdentitydense_6/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????<?
NoOpNoOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????00: : : : : : : : : : 2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
??
?/
"__inference__traced_restore_383247
file_prefix$
assignvariableop_adam_iter:	 (
assignvariableop_1_adam_beta_1: (
assignvariableop_2_adam_beta_2: '
assignvariableop_3_adam_decay: /
%assignvariableop_4_adam_learning_rate: =
#assignvariableop_5_conv2d_21_kernel:/
!assignvariableop_6_conv2d_21_bias:=
#assignvariableop_7_conv2d_22_kernel:/
!assignvariableop_8_conv2d_22_bias:=
#assignvariableop_9_conv2d_23_kernel:(0
"assignvariableop_10_conv2d_23_bias:(>
$assignvariableop_11_conv2d_24_kernel:((0
"assignvariableop_12_conv2d_24_bias:(5
"assignvariableop_13_dense_6_kernel:	?-<.
 assignvariableop_14_dense_6_bias:<5
"assignvariableop_15_dense_7_kernel:	<?-/
 assignvariableop_16_dense_7_bias:	?-G
-assignvariableop_17_conv2d_transpose_6_kernel:((9
+assignvariableop_18_conv2d_transpose_6_bias:(>
$assignvariableop_19_conv2d_25_kernel:((0
"assignvariableop_20_conv2d_25_bias:(G
-assignvariableop_21_conv2d_transpose_7_kernel:(9
+assignvariableop_22_conv2d_transpose_7_bias:>
$assignvariableop_23_conv2d_26_kernel:0
"assignvariableop_24_conv2d_26_bias:>
$assignvariableop_25_conv2d_27_kernel:0
"assignvariableop_26_conv2d_27_bias:#
assignvariableop_27_total: #
assignvariableop_28_count: E
+assignvariableop_29_adam_conv2d_21_kernel_m:7
)assignvariableop_30_adam_conv2d_21_bias_m:E
+assignvariableop_31_adam_conv2d_22_kernel_m:7
)assignvariableop_32_adam_conv2d_22_bias_m:E
+assignvariableop_33_adam_conv2d_23_kernel_m:(7
)assignvariableop_34_adam_conv2d_23_bias_m:(E
+assignvariableop_35_adam_conv2d_24_kernel_m:((7
)assignvariableop_36_adam_conv2d_24_bias_m:(<
)assignvariableop_37_adam_dense_6_kernel_m:	?-<5
'assignvariableop_38_adam_dense_6_bias_m:<<
)assignvariableop_39_adam_dense_7_kernel_m:	<?-6
'assignvariableop_40_adam_dense_7_bias_m:	?-N
4assignvariableop_41_adam_conv2d_transpose_6_kernel_m:((@
2assignvariableop_42_adam_conv2d_transpose_6_bias_m:(E
+assignvariableop_43_adam_conv2d_25_kernel_m:((7
)assignvariableop_44_adam_conv2d_25_bias_m:(N
4assignvariableop_45_adam_conv2d_transpose_7_kernel_m:(@
2assignvariableop_46_adam_conv2d_transpose_7_bias_m:E
+assignvariableop_47_adam_conv2d_26_kernel_m:7
)assignvariableop_48_adam_conv2d_26_bias_m:E
+assignvariableop_49_adam_conv2d_27_kernel_m:7
)assignvariableop_50_adam_conv2d_27_bias_m:E
+assignvariableop_51_adam_conv2d_21_kernel_v:7
)assignvariableop_52_adam_conv2d_21_bias_v:E
+assignvariableop_53_adam_conv2d_22_kernel_v:7
)assignvariableop_54_adam_conv2d_22_bias_v:E
+assignvariableop_55_adam_conv2d_23_kernel_v:(7
)assignvariableop_56_adam_conv2d_23_bias_v:(E
+assignvariableop_57_adam_conv2d_24_kernel_v:((7
)assignvariableop_58_adam_conv2d_24_bias_v:(<
)assignvariableop_59_adam_dense_6_kernel_v:	?-<5
'assignvariableop_60_adam_dense_6_bias_v:<<
)assignvariableop_61_adam_dense_7_kernel_v:	<?-6
'assignvariableop_62_adam_dense_7_bias_v:	?-N
4assignvariableop_63_adam_conv2d_transpose_6_kernel_v:((@
2assignvariableop_64_adam_conv2d_transpose_6_bias_v:(E
+assignvariableop_65_adam_conv2d_25_kernel_v:((7
)assignvariableop_66_adam_conv2d_25_bias_v:(N
4assignvariableop_67_adam_conv2d_transpose_7_kernel_v:(@
2assignvariableop_68_adam_conv2d_transpose_7_bias_v:E
+assignvariableop_69_adam_conv2d_26_kernel_v:7
)assignvariableop_70_adam_conv2d_26_bias_v:E
+assignvariableop_71_adam_conv2d_27_kernel_v:7
)assignvariableop_72_adam_conv2d_27_bias_v:
identity_74??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_8?AssignVariableOp_9?"
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*?!
value?!B?!JB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*?
value?B?JB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*X
dtypesN
L2J	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp#assignvariableop_5_conv2d_21_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_conv2d_21_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp#assignvariableop_7_conv2d_22_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_conv2d_22_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp#assignvariableop_9_conv2d_23_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_conv2d_23_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp$assignvariableop_11_conv2d_24_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp"assignvariableop_12_conv2d_24_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_dense_6_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp assignvariableop_14_dense_6_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp"assignvariableop_15_dense_7_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp assignvariableop_16_dense_7_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp-assignvariableop_17_conv2d_transpose_6_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp+assignvariableop_18_conv2d_transpose_6_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp$assignvariableop_19_conv2d_25_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp"assignvariableop_20_conv2d_25_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp-assignvariableop_21_conv2d_transpose_7_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp+assignvariableop_22_conv2d_transpose_7_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp$assignvariableop_23_conv2d_26_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp"assignvariableop_24_conv2d_26_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp$assignvariableop_25_conv2d_27_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp"assignvariableop_26_conv2d_27_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_21_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_21_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_22_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_22_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_23_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_23_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_24_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_24_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_dense_6_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_dense_6_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_dense_7_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp'assignvariableop_40_adam_dense_7_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp4assignvariableop_41_adam_conv2d_transpose_6_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp2assignvariableop_42_adam_conv2d_transpose_6_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_25_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_25_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp4assignvariableop_45_adam_conv2d_transpose_7_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp2assignvariableop_46_adam_conv2d_transpose_7_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_26_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_26_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_conv2d_27_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_conv2d_27_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_conv2d_21_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_conv2d_21_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp+assignvariableop_53_adam_conv2d_22_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp)assignvariableop_54_adam_conv2d_22_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_conv2d_23_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_conv2d_23_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_conv2d_24_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adam_conv2d_24_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp)assignvariableop_59_adam_dense_6_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp'assignvariableop_60_adam_dense_6_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp)assignvariableop_61_adam_dense_7_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp'assignvariableop_62_adam_dense_7_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp4assignvariableop_63_adam_conv2d_transpose_6_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp2assignvariableop_64_adam_conv2d_transpose_6_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp+assignvariableop_65_adam_conv2d_25_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_conv2d_25_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp4assignvariableop_67_adam_conv2d_transpose_7_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp2assignvariableop_68_adam_conv2d_transpose_7_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp+assignvariableop_69_adam_conv2d_26_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_conv2d_26_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOp+assignvariableop_71_adam_conv2d_27_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOp)assignvariableop_72_adam_conv2d_27_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_73Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_74IdentityIdentity_73:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_74Identity_74:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
O
encoder_input>
serving_default_encoder_input:0?????????00C
model_78
StatefulPartitionedCall:0?????????00tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
layer_with_weights-4
layer-8
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_network
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
 layer_with_weights-3
 layer-5
!layer_with_weights-4
!layer-6
"layer_with_weights-5
"layer-7
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses"
_tf_keras_network
?
)iter

*beta_1

+beta_2
	,decay
-learning_rate.m?/m?0m?1m?2m?3m?4m?5m?6m?7m?8m?9m?:m?;m?<m?=m?>m??m?@m?Am?Bm?Cm?.v?/v?0v?1v?2v?3v?4v?5v?6v?7v?8v?9v?:v?;v?<v?=v?>v??v?@v?Av?Bv?Cv?"
	optimizer
?
.0
/1
02
13
24
35
46
57
68
79
810
911
:12
;13
<14
=15
>16
?17
@18
A19
B20
C21"
trackable_list_wrapper
?
.0
/1
02
13
24
35
46
57
68
79
810
911
:12
;13
<14
=15
>16
?17
@18
A19
B20
C21"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_autoencoder_layer_call_fn_381356
,__inference_autoencoder_layer_call_fn_381708
,__inference_autoencoder_layer_call_fn_381757
,__inference_autoencoder_layer_call_fn_381553?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_autoencoder_layer_call_and_return_conditional_losses_381876
G__inference_autoencoder_layer_call_and_return_conditional_losses_382009
G__inference_autoencoder_layer_call_and_return_conditional_losses_381603
G__inference_autoencoder_layer_call_and_return_conditional_losses_381653?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_380389encoder_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
Iserving_default"
signature_map
?

.kernel
/bias
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses"
_tf_keras_layer
?

0kernel
1bias
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses"
_tf_keras_layer
?
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z_random_generator
[__call__
*\&call_and_return_all_conditional_losses"
_tf_keras_layer
?

2kernel
3bias
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses"
_tf_keras_layer
?

4kernel
5bias
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses"
_tf_keras_layer
?
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m_random_generator
n__call__
*o&call_and_return_all_conditional_losses"
_tf_keras_layer
?
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses"
_tf_keras_layer
?

6kernel
7bias
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
f
.0
/1
02
13
24
35
46
57
68
79"
trackable_list_wrapper
f
.0
/1
02
13
24
35
46
57
68
79"
trackable_list_wrapper
 "
trackable_list_wrapper
?
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_model_6_layer_call_fn_380526
(__inference_model_6_layer_call_fn_382085
(__inference_model_6_layer_call_fn_382110
(__inference_model_6_layer_call_fn_380735?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_model_6_layer_call_and_return_conditional_losses_382152
C__inference_model_6_layer_call_and_return_conditional_losses_382208
C__inference_model_6_layer_call_and_return_conditional_losses_380767
C__inference_model_6_layer_call_and_return_conditional_losses_380799?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
"
_tf_keras_input_layer
?

8kernel
9bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

:kernel
;bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

<kernel
=bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

>kernel
?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

@kernel
Abias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Bkernel
Cbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
v
80
91
:2
;3
<4
=5
>6
?7
@8
A9
B10
C11"
trackable_list_wrapper
v
80
91
:2
;3
<4
=5
>6
?7
@8
A9
B10
C11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_model_7_layer_call_fn_381017
(__inference_model_7_layer_call_fn_382237
(__inference_model_7_layer_call_fn_382266
(__inference_model_7_layer_call_fn_381185?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_model_7_layer_call_and_return_conditional_losses_382347
C__inference_model_7_layer_call_and_return_conditional_losses_382428
C__inference_model_7_layer_call_and_return_conditional_losses_381220
C__inference_model_7_layer_call_and_return_conditional_losses_381255?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
*:(2conv2d_21/kernel
:2conv2d_21/bias
*:(2conv2d_22/kernel
:2conv2d_22/bias
*:((2conv2d_23/kernel
:(2conv2d_23/bias
*:(((2conv2d_24/kernel
:(2conv2d_24/bias
!:	?-<2dense_6/kernel
:<2dense_6/bias
!:	<?-2dense_7/kernel
:?-2dense_7/bias
3:1((2conv2d_transpose_6/kernel
%:#(2conv2d_transpose_6/bias
*:(((2conv2d_25/kernel
:(2conv2d_25/bias
3:1(2conv2d_transpose_7/kernel
%:#2conv2d_transpose_7/bias
*:(2conv2d_26/kernel
:2conv2d_26/bias
*:(2conv2d_27/kernel
:2conv2d_27/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
$__inference_signature_wrapper_382060encoder_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_conv2d_21_layer_call_fn_382437?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_21_layer_call_and_return_conditional_losses_382448?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_conv2d_22_layer_call_fn_382457?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_22_layer_call_and_return_conditional_losses_382468?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2?
*__inference_dropout_6_layer_call_fn_382473
*__inference_dropout_6_layer_call_fn_382478?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_6_layer_call_and_return_conditional_losses_382483
E__inference_dropout_6_layer_call_and_return_conditional_losses_382495?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_conv2d_23_layer_call_fn_382504?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_23_layer_call_and_return_conditional_losses_382515?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_conv2d_24_layer_call_fn_382524?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_24_layer_call_and_return_conditional_losses_382535?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
i	variables
jtrainable_variables
kregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2?
*__inference_dropout_7_layer_call_fn_382540
*__inference_dropout_7_layer_call_fn_382545?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_7_layer_call_and_return_conditional_losses_382550
E__inference_dropout_7_layer_call_and_return_conditional_losses_382562?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_flatten_3_layer_call_fn_382567?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_3_layer_call_and_return_conditional_losses_382573?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_dense_6_layer_call_fn_382582?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_6_layer_call_and_return_conditional_losses_382592?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_dense_7_layer_call_fn_382601?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_7_layer_call_and_return_conditional_losses_382611?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_reshape_3_layer_call_fn_382616?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_reshape_3_layer_call_and_return_conditional_losses_382630?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
3__inference_conv2d_transpose_6_layer_call_fn_382639?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_382673?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_conv2d_25_layer_call_fn_382682?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_25_layer_call_and_return_conditional_losses_382693?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
3__inference_conv2d_transpose_7_layer_call_fn_382702?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_382736?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_conv2d_26_layer_call_fn_382745?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_26_layer_call_and_return_conditional_losses_382756?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_conv2d_27_layer_call_fn_382765?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_27_layer_call_and_return_conditional_losses_382776?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
X
0
1
2
3
4
 5
!6
"7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
/:-2Adam/conv2d_21/kernel/m
!:2Adam/conv2d_21/bias/m
/:-2Adam/conv2d_22/kernel/m
!:2Adam/conv2d_22/bias/m
/:-(2Adam/conv2d_23/kernel/m
!:(2Adam/conv2d_23/bias/m
/:-((2Adam/conv2d_24/kernel/m
!:(2Adam/conv2d_24/bias/m
&:$	?-<2Adam/dense_6/kernel/m
:<2Adam/dense_6/bias/m
&:$	<?-2Adam/dense_7/kernel/m
 :?-2Adam/dense_7/bias/m
8:6((2 Adam/conv2d_transpose_6/kernel/m
*:((2Adam/conv2d_transpose_6/bias/m
/:-((2Adam/conv2d_25/kernel/m
!:(2Adam/conv2d_25/bias/m
8:6(2 Adam/conv2d_transpose_7/kernel/m
*:(2Adam/conv2d_transpose_7/bias/m
/:-2Adam/conv2d_26/kernel/m
!:2Adam/conv2d_26/bias/m
/:-2Adam/conv2d_27/kernel/m
!:2Adam/conv2d_27/bias/m
/:-2Adam/conv2d_21/kernel/v
!:2Adam/conv2d_21/bias/v
/:-2Adam/conv2d_22/kernel/v
!:2Adam/conv2d_22/bias/v
/:-(2Adam/conv2d_23/kernel/v
!:(2Adam/conv2d_23/bias/v
/:-((2Adam/conv2d_24/kernel/v
!:(2Adam/conv2d_24/bias/v
&:$	?-<2Adam/dense_6/kernel/v
:<2Adam/dense_6/bias/v
&:$	<?-2Adam/dense_7/kernel/v
 :?-2Adam/dense_7/bias/v
8:6((2 Adam/conv2d_transpose_6/kernel/v
*:((2Adam/conv2d_transpose_6/bias/v
/:-((2Adam/conv2d_25/kernel/v
!:(2Adam/conv2d_25/bias/v
8:6(2 Adam/conv2d_transpose_7/kernel/v
*:(2Adam/conv2d_transpose_7/bias/v
/:-2Adam/conv2d_26/kernel/v
!:2Adam/conv2d_26/bias/v
/:-2Adam/conv2d_27/kernel/v
!:2Adam/conv2d_27/bias/v?
!__inference__wrapped_model_380389?./0123456789:;<=>?@ABC>?;
4?1
/?,
encoder_input?????????00
? "9?6
4
model_7)?&
model_7?????????00?
G__inference_autoencoder_layer_call_and_return_conditional_losses_381603?./0123456789:;<=>?@ABCF?C
<?9
/?,
encoder_input?????????00
p 

 
? "-?*
#? 
0?????????00
? ?
G__inference_autoencoder_layer_call_and_return_conditional_losses_381653?./0123456789:;<=>?@ABCF?C
<?9
/?,
encoder_input?????????00
p

 
? "-?*
#? 
0?????????00
? ?
G__inference_autoencoder_layer_call_and_return_conditional_losses_381876?./0123456789:;<=>?@ABC??<
5?2
(?%
inputs?????????00
p 

 
? "-?*
#? 
0?????????00
? ?
G__inference_autoencoder_layer_call_and_return_conditional_losses_382009?./0123456789:;<=>?@ABC??<
5?2
(?%
inputs?????????00
p

 
? "-?*
#? 
0?????????00
? ?
,__inference_autoencoder_layer_call_fn_381356?./0123456789:;<=>?@ABCF?C
<?9
/?,
encoder_input?????????00
p 

 
? " ??????????00?
,__inference_autoencoder_layer_call_fn_381553?./0123456789:;<=>?@ABCF?C
<?9
/?,
encoder_input?????????00
p

 
? " ??????????00?
,__inference_autoencoder_layer_call_fn_381708{./0123456789:;<=>?@ABC??<
5?2
(?%
inputs?????????00
p 

 
? " ??????????00?
,__inference_autoencoder_layer_call_fn_381757{./0123456789:;<=>?@ABC??<
5?2
(?%
inputs?????????00
p

 
? " ??????????00?
E__inference_conv2d_21_layer_call_and_return_conditional_losses_382448l./7?4
-?*
(?%
inputs?????????00
? "-?*
#? 
0?????????00
? ?
*__inference_conv2d_21_layer_call_fn_382437_./7?4
-?*
(?%
inputs?????????00
? " ??????????00?
E__inference_conv2d_22_layer_call_and_return_conditional_losses_382468l017?4
-?*
(?%
inputs?????????00
? "-?*
#? 
0?????????
? ?
*__inference_conv2d_22_layer_call_fn_382457_017?4
-?*
(?%
inputs?????????00
? " ???????????
E__inference_conv2d_23_layer_call_and_return_conditional_losses_382515l237?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????(
? ?
*__inference_conv2d_23_layer_call_fn_382504_237?4
-?*
(?%
inputs?????????
? " ??????????(?
E__inference_conv2d_24_layer_call_and_return_conditional_losses_382535l457?4
-?*
(?%
inputs?????????(
? "-?*
#? 
0?????????(
? ?
*__inference_conv2d_24_layer_call_fn_382524_457?4
-?*
(?%
inputs?????????(
? " ??????????(?
E__inference_conv2d_25_layer_call_and_return_conditional_losses_382693l<=7?4
-?*
(?%
inputs?????????(
? "-?*
#? 
0?????????(
? ?
*__inference_conv2d_25_layer_call_fn_382682_<=7?4
-?*
(?%
inputs?????????(
? " ??????????(?
E__inference_conv2d_26_layer_call_and_return_conditional_losses_382756l@A7?4
-?*
(?%
inputs?????????00
? "-?*
#? 
0?????????00
? ?
*__inference_conv2d_26_layer_call_fn_382745_@A7?4
-?*
(?%
inputs?????????00
? " ??????????00?
E__inference_conv2d_27_layer_call_and_return_conditional_losses_382776lBC7?4
-?*
(?%
inputs?????????00
? "-?*
#? 
0?????????00
? ?
*__inference_conv2d_27_layer_call_fn_382765_BC7?4
-?*
(?%
inputs?????????00
? " ??????????00?
N__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_382673?:;I?F
??<
:?7
inputs+???????????????????????????(
? "??<
5?2
0+???????????????????????????(
? ?
3__inference_conv2d_transpose_6_layer_call_fn_382639?:;I?F
??<
:?7
inputs+???????????????????????????(
? "2?/+???????????????????????????(?
N__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_382736?>?I?F
??<
:?7
inputs+???????????????????????????(
? "??<
5?2
0+???????????????????????????
? ?
3__inference_conv2d_transpose_7_layer_call_fn_382702?>?I?F
??<
:?7
inputs+???????????????????????????(
? "2?/+????????????????????????????
C__inference_dense_6_layer_call_and_return_conditional_losses_382592]670?-
&?#
!?
inputs??????????-
? "%?"
?
0?????????<
? |
(__inference_dense_6_layer_call_fn_382582P670?-
&?#
!?
inputs??????????-
? "??????????<?
C__inference_dense_7_layer_call_and_return_conditional_losses_382611]89/?,
%?"
 ?
inputs?????????<
? "&?#
?
0??????????-
? |
(__inference_dense_7_layer_call_fn_382601P89/?,
%?"
 ?
inputs?????????<
? "???????????-?
E__inference_dropout_6_layer_call_and_return_conditional_losses_382483l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
E__inference_dropout_6_layer_call_and_return_conditional_losses_382495l;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
*__inference_dropout_6_layer_call_fn_382473_;?8
1?.
(?%
inputs?????????
p 
? " ???????????
*__inference_dropout_6_layer_call_fn_382478_;?8
1?.
(?%
inputs?????????
p
? " ???????????
E__inference_dropout_7_layer_call_and_return_conditional_losses_382550l;?8
1?.
(?%
inputs?????????(
p 
? "-?*
#? 
0?????????(
? ?
E__inference_dropout_7_layer_call_and_return_conditional_losses_382562l;?8
1?.
(?%
inputs?????????(
p
? "-?*
#? 
0?????????(
? ?
*__inference_dropout_7_layer_call_fn_382540_;?8
1?.
(?%
inputs?????????(
p 
? " ??????????(?
*__inference_dropout_7_layer_call_fn_382545_;?8
1?.
(?%
inputs?????????(
p
? " ??????????(?
E__inference_flatten_3_layer_call_and_return_conditional_losses_382573a7?4
-?*
(?%
inputs?????????(
? "&?#
?
0??????????-
? ?
*__inference_flatten_3_layer_call_fn_382567T7?4
-?*
(?%
inputs?????????(
? "???????????-?
C__inference_model_6_layer_call_and_return_conditional_losses_380767{
./01234567F?C
<?9
/?,
encoder_input?????????00
p 

 
? "%?"
?
0?????????<
? ?
C__inference_model_6_layer_call_and_return_conditional_losses_380799{
./01234567F?C
<?9
/?,
encoder_input?????????00
p

 
? "%?"
?
0?????????<
? ?
C__inference_model_6_layer_call_and_return_conditional_losses_382152t
./01234567??<
5?2
(?%
inputs?????????00
p 

 
? "%?"
?
0?????????<
? ?
C__inference_model_6_layer_call_and_return_conditional_losses_382208t
./01234567??<
5?2
(?%
inputs?????????00
p

 
? "%?"
?
0?????????<
? ?
(__inference_model_6_layer_call_fn_380526n
./01234567F?C
<?9
/?,
encoder_input?????????00
p 

 
? "??????????<?
(__inference_model_6_layer_call_fn_380735n
./01234567F?C
<?9
/?,
encoder_input?????????00
p

 
? "??????????<?
(__inference_model_6_layer_call_fn_382085g
./01234567??<
5?2
(?%
inputs?????????00
p 

 
? "??????????<?
(__inference_model_6_layer_call_fn_382110g
./01234567??<
5?2
(?%
inputs?????????00
p

 
? "??????????<?
C__inference_model_7_layer_call_and_return_conditional_losses_381220w89:;<=>?@ABC8?5
.?+
!?
input_4?????????<
p 

 
? "-?*
#? 
0?????????00
? ?
C__inference_model_7_layer_call_and_return_conditional_losses_381255w89:;<=>?@ABC8?5
.?+
!?
input_4?????????<
p

 
? "-?*
#? 
0?????????00
? ?
C__inference_model_7_layer_call_and_return_conditional_losses_382347v89:;<=>?@ABC7?4
-?*
 ?
inputs?????????<
p 

 
? "-?*
#? 
0?????????00
? ?
C__inference_model_7_layer_call_and_return_conditional_losses_382428v89:;<=>?@ABC7?4
-?*
 ?
inputs?????????<
p

 
? "-?*
#? 
0?????????00
? ?
(__inference_model_7_layer_call_fn_381017j89:;<=>?@ABC8?5
.?+
!?
input_4?????????<
p 

 
? " ??????????00?
(__inference_model_7_layer_call_fn_381185j89:;<=>?@ABC8?5
.?+
!?
input_4?????????<
p

 
? " ??????????00?
(__inference_model_7_layer_call_fn_382237i89:;<=>?@ABC7?4
-?*
 ?
inputs?????????<
p 

 
? " ??????????00?
(__inference_model_7_layer_call_fn_382266i89:;<=>?@ABC7?4
-?*
 ?
inputs?????????<
p

 
? " ??????????00?
E__inference_reshape_3_layer_call_and_return_conditional_losses_382630a0?-
&?#
!?
inputs??????????-
? "-?*
#? 
0?????????(
? ?
*__inference_reshape_3_layer_call_fn_382616T0?-
&?#
!?
inputs??????????-
? " ??????????(?
$__inference_signature_wrapper_382060?./0123456789:;<=>?@ABCO?L
? 
E?B
@
encoder_input/?,
encoder_input?????????00"9?6
4
model_7)?&
model_7?????????00