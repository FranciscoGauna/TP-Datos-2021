??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
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
E
Relu
features"T
activations"T"
Ttype:
2	
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
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
{
dense_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	A?* 
shared_namedense_41/kernel
t
#dense_41/kernel/Read/ReadVariableOpReadVariableOpdense_41/kernel*
_output_shapes
:	A?*
dtype0
s
dense_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_41/bias
l
!dense_41/bias/Read/ReadVariableOpReadVariableOpdense_41/bias*
_output_shapes	
:?*
dtype0
|
dense_42/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_42/kernel
u
#dense_42/kernel/Read/ReadVariableOpReadVariableOpdense_42/kernel* 
_output_shapes
:
??*
dtype0
s
dense_42/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_42/bias
l
!dense_42/bias/Read/ReadVariableOpReadVariableOpdense_42/bias*
_output_shapes	
:?*
dtype0
|
dense_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_43/kernel
u
#dense_43/kernel/Read/ReadVariableOpReadVariableOpdense_43/kernel* 
_output_shapes
:
??*
dtype0
s
dense_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_43/bias
l
!dense_43/bias/Read/ReadVariableOpReadVariableOpdense_43/bias*
_output_shapes	
:?*
dtype0
|
dense_44/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_44/kernel
u
#dense_44/kernel/Read/ReadVariableOpReadVariableOpdense_44/kernel* 
_output_shapes
:
??*
dtype0
s
dense_44/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_44/bias
l
!dense_44/bias/Read/ReadVariableOpReadVariableOpdense_44/bias*
_output_shapes	
:?*
dtype0
{
dense_45/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_45/kernel
t
#dense_45/kernel/Read/ReadVariableOpReadVariableOpdense_45/kernel*
_output_shapes
:	?*
dtype0
r
dense_45/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_45/bias
k
!dense_45/bias/Read/ReadVariableOpReadVariableOpdense_45/bias*
_output_shapes
:*
dtype0
l
Adagrad/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameAdagrad/iter
e
 Adagrad/iter/Read/ReadVariableOpReadVariableOpAdagrad/iter*
_output_shapes
: *
dtype0	
n
Adagrad/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdagrad/decay
g
!Adagrad/decay/Read/ReadVariableOpReadVariableOpAdagrad/decay*
_output_shapes
: *
dtype0
~
Adagrad/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdagrad/learning_rate
w
)Adagrad/learning_rate/Read/ReadVariableOpReadVariableOpAdagrad/learning_rate*
_output_shapes
: *
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
#Adagrad/dense_41/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:	A?*4
shared_name%#Adagrad/dense_41/kernel/accumulator
?
7Adagrad/dense_41/kernel/accumulator/Read/ReadVariableOpReadVariableOp#Adagrad/dense_41/kernel/accumulator*
_output_shapes
:	A?*
dtype0
?
!Adagrad/dense_41/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adagrad/dense_41/bias/accumulator
?
5Adagrad/dense_41/bias/accumulator/Read/ReadVariableOpReadVariableOp!Adagrad/dense_41/bias/accumulator*
_output_shapes	
:?*
dtype0
?
#Adagrad/dense_42/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*4
shared_name%#Adagrad/dense_42/kernel/accumulator
?
7Adagrad/dense_42/kernel/accumulator/Read/ReadVariableOpReadVariableOp#Adagrad/dense_42/kernel/accumulator* 
_output_shapes
:
??*
dtype0
?
!Adagrad/dense_42/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adagrad/dense_42/bias/accumulator
?
5Adagrad/dense_42/bias/accumulator/Read/ReadVariableOpReadVariableOp!Adagrad/dense_42/bias/accumulator*
_output_shapes	
:?*
dtype0
?
#Adagrad/dense_43/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*4
shared_name%#Adagrad/dense_43/kernel/accumulator
?
7Adagrad/dense_43/kernel/accumulator/Read/ReadVariableOpReadVariableOp#Adagrad/dense_43/kernel/accumulator* 
_output_shapes
:
??*
dtype0
?
!Adagrad/dense_43/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adagrad/dense_43/bias/accumulator
?
5Adagrad/dense_43/bias/accumulator/Read/ReadVariableOpReadVariableOp!Adagrad/dense_43/bias/accumulator*
_output_shapes	
:?*
dtype0
?
#Adagrad/dense_44/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*4
shared_name%#Adagrad/dense_44/kernel/accumulator
?
7Adagrad/dense_44/kernel/accumulator/Read/ReadVariableOpReadVariableOp#Adagrad/dense_44/kernel/accumulator* 
_output_shapes
:
??*
dtype0
?
!Adagrad/dense_44/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adagrad/dense_44/bias/accumulator
?
5Adagrad/dense_44/bias/accumulator/Read/ReadVariableOpReadVariableOp!Adagrad/dense_44/bias/accumulator*
_output_shapes	
:?*
dtype0
?
#Adagrad/dense_45/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*4
shared_name%#Adagrad/dense_45/kernel/accumulator
?
7Adagrad/dense_45/kernel/accumulator/Read/ReadVariableOpReadVariableOp#Adagrad/dense_45/kernel/accumulator*
_output_shapes
:	?*
dtype0
?
!Adagrad/dense_45/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adagrad/dense_45/bias/accumulator
?
5Adagrad/dense_45/bias/accumulator/Read/ReadVariableOpReadVariableOp!Adagrad/dense_45/bias/accumulator*
_output_shapes
:*
dtype0

NoOpNoOp
?-
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?-
value?,B?, B?,
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
h

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
?
*iter
	+decay
,learning_rateaccumulatorVaccumulatorWaccumulatorXaccumulatorYaccumulatorZaccumulator[accumulator\accumulator]$accumulator^%accumulator_
F
0
1
2
3
4
5
6
7
$8
%9
F
0
1
2
3
4
5
6
7
$8
%9
 
?
-metrics
	variables
.layer_regularization_losses
/non_trainable_variables
trainable_variables
0layer_metrics

1layers
	regularization_losses
 
[Y
VARIABLE_VALUEdense_41/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_41/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
2metrics
	variables
3layer_regularization_losses
4non_trainable_variables
trainable_variables
5layer_metrics

6layers
regularization_losses
[Y
VARIABLE_VALUEdense_42/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_42/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
7metrics
	variables
8layer_regularization_losses
9non_trainable_variables
trainable_variables
:layer_metrics

;layers
regularization_losses
[Y
VARIABLE_VALUEdense_43/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_43/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
<metrics
	variables
=layer_regularization_losses
>non_trainable_variables
trainable_variables
?layer_metrics

@layers
regularization_losses
[Y
VARIABLE_VALUEdense_44/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_44/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Ametrics
 	variables
Blayer_regularization_losses
Cnon_trainable_variables
!trainable_variables
Dlayer_metrics

Elayers
"regularization_losses
[Y
VARIABLE_VALUEdense_45/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_45/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1

$0
%1
 
?
Fmetrics
&	variables
Glayer_regularization_losses
Hnon_trainable_variables
'trainable_variables
Ilayer_metrics

Jlayers
(regularization_losses
KI
VARIABLE_VALUEAdagrad/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEAdagrad/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEAdagrad/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

K0
L1
 
 
 
#
0
1
2
3
4
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Mtotal
	Ncount
O	variables
P	keras_api
D
	Qtotal
	Rcount
S
_fn_kwargs
T	variables
U	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

M0
N1

O	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

Q0
R1

T	variables
??
VARIABLE_VALUE#Adagrad/dense_41/kernel/accumulator\layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adagrad/dense_41/bias/accumulatorZlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adagrad/dense_42/kernel/accumulator\layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adagrad/dense_42/bias/accumulatorZlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adagrad/dense_43/kernel/accumulator\layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adagrad/dense_43/bias/accumulatorZlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adagrad/dense_44/kernel/accumulator\layer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adagrad/dense_44/bias/accumulatorZlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adagrad/dense_45/kernel/accumulator\layer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adagrad/dense_45/bias/accumulatorZlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_dense_41_inputPlaceholder*'
_output_shapes
:?????????A*
dtype0*
shape:?????????A
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_41_inputdense_41/kerneldense_41/biasdense_42/kerneldense_42/biasdense_43/kerneldense_43/biasdense_44/kerneldense_44/biasdense_45/kerneldense_45/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_signature_wrapper_22988555
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_41/kernel/Read/ReadVariableOp!dense_41/bias/Read/ReadVariableOp#dense_42/kernel/Read/ReadVariableOp!dense_42/bias/Read/ReadVariableOp#dense_43/kernel/Read/ReadVariableOp!dense_43/bias/Read/ReadVariableOp#dense_44/kernel/Read/ReadVariableOp!dense_44/bias/Read/ReadVariableOp#dense_45/kernel/Read/ReadVariableOp!dense_45/bias/Read/ReadVariableOp Adagrad/iter/Read/ReadVariableOp!Adagrad/decay/Read/ReadVariableOp)Adagrad/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp7Adagrad/dense_41/kernel/accumulator/Read/ReadVariableOp5Adagrad/dense_41/bias/accumulator/Read/ReadVariableOp7Adagrad/dense_42/kernel/accumulator/Read/ReadVariableOp5Adagrad/dense_42/bias/accumulator/Read/ReadVariableOp7Adagrad/dense_43/kernel/accumulator/Read/ReadVariableOp5Adagrad/dense_43/bias/accumulator/Read/ReadVariableOp7Adagrad/dense_44/kernel/accumulator/Read/ReadVariableOp5Adagrad/dense_44/bias/accumulator/Read/ReadVariableOp7Adagrad/dense_45/kernel/accumulator/Read/ReadVariableOp5Adagrad/dense_45/bias/accumulator/Read/ReadVariableOpConst*(
Tin!
2	*
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
GPU2*0J 8? **
f%R#
!__inference__traced_save_22988890
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_41/kerneldense_41/biasdense_42/kerneldense_42/biasdense_43/kerneldense_43/biasdense_44/kerneldense_44/biasdense_45/kerneldense_45/biasAdagrad/iterAdagrad/decayAdagrad/learning_ratetotalcounttotal_1count_1#Adagrad/dense_41/kernel/accumulator!Adagrad/dense_41/bias/accumulator#Adagrad/dense_42/kernel/accumulator!Adagrad/dense_42/bias/accumulator#Adagrad/dense_43/kernel/accumulator!Adagrad/dense_43/bias/accumulator#Adagrad/dense_44/kernel/accumulator!Adagrad/dense_44/bias/accumulator#Adagrad/dense_45/kernel/accumulator!Adagrad/dense_45/bias/accumulator*'
Tin 
2*
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
GPU2*0J 8? *-
f(R&
$__inference__traced_restore_22988981??
?
?
J__inference_sequential_9_layer_call_and_return_conditional_losses_22988291

inputs$
dense_41_22988217:	A? 
dense_41_22988219:	?%
dense_42_22988234:
?? 
dense_42_22988236:	?%
dense_43_22988251:
?? 
dense_43_22988253:	?%
dense_44_22988268:
?? 
dense_44_22988270:	?$
dense_45_22988285:	?
dense_45_22988287:
identity?? dense_41/StatefulPartitionedCall? dense_42/StatefulPartitionedCall? dense_43/StatefulPartitionedCall? dense_44/StatefulPartitionedCall? dense_45/StatefulPartitionedCall?
 dense_41/StatefulPartitionedCallStatefulPartitionedCallinputsdense_41_22988217dense_41_22988219*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_41_layer_call_and_return_conditional_losses_229882162"
 dense_41/StatefulPartitionedCall?
 dense_42/StatefulPartitionedCallStatefulPartitionedCall)dense_41/StatefulPartitionedCall:output:0dense_42_22988234dense_42_22988236*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_42_layer_call_and_return_conditional_losses_229882332"
 dense_42/StatefulPartitionedCall?
 dense_43/StatefulPartitionedCallStatefulPartitionedCall)dense_42/StatefulPartitionedCall:output:0dense_43_22988251dense_43_22988253*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_43_layer_call_and_return_conditional_losses_229882502"
 dense_43/StatefulPartitionedCall?
 dense_44/StatefulPartitionedCallStatefulPartitionedCall)dense_43/StatefulPartitionedCall:output:0dense_44_22988268dense_44_22988270*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_44_layer_call_and_return_conditional_losses_229882672"
 dense_44/StatefulPartitionedCall?
 dense_45/StatefulPartitionedCallStatefulPartitionedCall)dense_44/StatefulPartitionedCall:output:0dense_45_22988285dense_45_22988287*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_45_layer_call_and_return_conditional_losses_229882842"
 dense_45/StatefulPartitionedCall?
IdentityIdentity)dense_45/StatefulPartitionedCall:output:0!^dense_41/StatefulPartitionedCall!^dense_42/StatefulPartitionedCall!^dense_43/StatefulPartitionedCall!^dense_44/StatefulPartitionedCall!^dense_45/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????A: : : : : : : : : : 2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2D
 dense_42/StatefulPartitionedCall dense_42/StatefulPartitionedCall2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall2D
 dense_44/StatefulPartitionedCall dense_44/StatefulPartitionedCall2D
 dense_45/StatefulPartitionedCall dense_45/StatefulPartitionedCall:O K
'
_output_shapes
:?????????A
 
_user_specified_nameinputs
?
?
F__inference_dense_41_layer_call_and_return_conditional_losses_22988216

inputs1
matmul_readvariableop_resource:	A?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????A2
Cast?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	A?*
dtype02
MatMul/ReadVariableOpv
MatMulMatMulCast:y:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????A
 
_user_specified_nameinputs
?
?
+__inference_dense_41_layer_call_fn_22988694

inputs
unknown:	A?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_41_layer_call_and_return_conditional_losses_229882162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????A: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????A
 
_user_specified_nameinputs
?
?
+__inference_dense_45_layer_call_fn_22988775

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_45_layer_call_and_return_conditional_losses_229882842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_dense_42_layer_call_fn_22988715

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_42_layer_call_and_return_conditional_losses_229882332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_dense_43_layer_call_fn_22988735

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_43_layer_call_and_return_conditional_losses_229882502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_dense_44_layer_call_fn_22988755

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_44_layer_call_and_return_conditional_losses_229882672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
F__inference_dense_44_layer_call_and_return_conditional_losses_22988267

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
/__inference_sequential_9_layer_call_fn_22988605

inputs
unknown:	A?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_9_layer_call_and_return_conditional_losses_229884202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????A: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????A
 
_user_specified_nameinputs
?

?
F__inference_dense_42_layer_call_and_return_conditional_losses_22988726

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?@
?
!__inference__traced_save_22988890
file_prefix.
*savev2_dense_41_kernel_read_readvariableop,
(savev2_dense_41_bias_read_readvariableop.
*savev2_dense_42_kernel_read_readvariableop,
(savev2_dense_42_bias_read_readvariableop.
*savev2_dense_43_kernel_read_readvariableop,
(savev2_dense_43_bias_read_readvariableop.
*savev2_dense_44_kernel_read_readvariableop,
(savev2_dense_44_bias_read_readvariableop.
*savev2_dense_45_kernel_read_readvariableop,
(savev2_dense_45_bias_read_readvariableop+
'savev2_adagrad_iter_read_readvariableop	,
(savev2_adagrad_decay_read_readvariableop4
0savev2_adagrad_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopB
>savev2_adagrad_dense_41_kernel_accumulator_read_readvariableop@
<savev2_adagrad_dense_41_bias_accumulator_read_readvariableopB
>savev2_adagrad_dense_42_kernel_accumulator_read_readvariableop@
<savev2_adagrad_dense_42_bias_accumulator_read_readvariableopB
>savev2_adagrad_dense_43_kernel_accumulator_read_readvariableop@
<savev2_adagrad_dense_43_bias_accumulator_read_readvariableopB
>savev2_adagrad_dense_44_kernel_accumulator_read_readvariableop@
<savev2_adagrad_dense_44_bias_accumulator_read_readvariableopB
>savev2_adagrad_dense_45_kernel_accumulator_read_readvariableop@
<savev2_adagrad_dense_45_bias_accumulator_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_41_kernel_read_readvariableop(savev2_dense_41_bias_read_readvariableop*savev2_dense_42_kernel_read_readvariableop(savev2_dense_42_bias_read_readvariableop*savev2_dense_43_kernel_read_readvariableop(savev2_dense_43_bias_read_readvariableop*savev2_dense_44_kernel_read_readvariableop(savev2_dense_44_bias_read_readvariableop*savev2_dense_45_kernel_read_readvariableop(savev2_dense_45_bias_read_readvariableop'savev2_adagrad_iter_read_readvariableop(savev2_adagrad_decay_read_readvariableop0savev2_adagrad_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop>savev2_adagrad_dense_41_kernel_accumulator_read_readvariableop<savev2_adagrad_dense_41_bias_accumulator_read_readvariableop>savev2_adagrad_dense_42_kernel_accumulator_read_readvariableop<savev2_adagrad_dense_42_bias_accumulator_read_readvariableop>savev2_adagrad_dense_43_kernel_accumulator_read_readvariableop<savev2_adagrad_dense_43_bias_accumulator_read_readvariableop>savev2_adagrad_dense_44_kernel_accumulator_read_readvariableop<savev2_adagrad_dense_44_bias_accumulator_read_readvariableop>savev2_adagrad_dense_45_kernel_accumulator_read_readvariableop<savev2_adagrad_dense_45_bias_accumulator_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 **
dtypes 
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	A?:?:
??:?:
??:?:
??:?:	?:: : : : : : : :	A?:?:
??:?:
??:?:
??:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	A?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%	!

_output_shapes
:	?: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	A?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: 
?	
?
&__inference_signature_wrapper_22988555
dense_41_input
unknown:	A?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_41_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__wrapped_model_229881972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????A: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????A
(
_user_specified_namedense_41_input
?2
?
J__inference_sequential_9_layer_call_and_return_conditional_losses_22988685

inputs:
'dense_41_matmul_readvariableop_resource:	A?7
(dense_41_biasadd_readvariableop_resource:	?;
'dense_42_matmul_readvariableop_resource:
??7
(dense_42_biasadd_readvariableop_resource:	?;
'dense_43_matmul_readvariableop_resource:
??7
(dense_43_biasadd_readvariableop_resource:	?;
'dense_44_matmul_readvariableop_resource:
??7
(dense_44_biasadd_readvariableop_resource:	?:
'dense_45_matmul_readvariableop_resource:	?6
(dense_45_biasadd_readvariableop_resource:
identity??dense_41/BiasAdd/ReadVariableOp?dense_41/MatMul/ReadVariableOp?dense_42/BiasAdd/ReadVariableOp?dense_42/MatMul/ReadVariableOp?dense_43/BiasAdd/ReadVariableOp?dense_43/MatMul/ReadVariableOp?dense_44/BiasAdd/ReadVariableOp?dense_44/MatMul/ReadVariableOp?dense_45/BiasAdd/ReadVariableOp?dense_45/MatMul/ReadVariableOpo
dense_41/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????A2
dense_41/Cast?
dense_41/MatMul/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource*
_output_shapes
:	A?*
dtype02 
dense_41/MatMul/ReadVariableOp?
dense_41/MatMulMatMuldense_41/Cast:y:0&dense_41/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_41/MatMul?
dense_41/BiasAdd/ReadVariableOpReadVariableOp(dense_41_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_41/BiasAdd/ReadVariableOp?
dense_41/BiasAddBiasAdddense_41/MatMul:product:0'dense_41/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_41/BiasAddt
dense_41/ReluReludense_41/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_41/Relu?
dense_42/MatMul/ReadVariableOpReadVariableOp'dense_42_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_42/MatMul/ReadVariableOp?
dense_42/MatMulMatMuldense_41/Relu:activations:0&dense_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_42/MatMul?
dense_42/BiasAdd/ReadVariableOpReadVariableOp(dense_42_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_42/BiasAdd/ReadVariableOp?
dense_42/BiasAddBiasAdddense_42/MatMul:product:0'dense_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_42/BiasAddt
dense_42/ReluReludense_42/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_42/Relu?
dense_43/MatMul/ReadVariableOpReadVariableOp'dense_43_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_43/MatMul/ReadVariableOp?
dense_43/MatMulMatMuldense_42/Relu:activations:0&dense_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_43/MatMul?
dense_43/BiasAdd/ReadVariableOpReadVariableOp(dense_43_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_43/BiasAdd/ReadVariableOp?
dense_43/BiasAddBiasAdddense_43/MatMul:product:0'dense_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_43/BiasAddt
dense_43/ReluReludense_43/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_43/Relu?
dense_44/MatMul/ReadVariableOpReadVariableOp'dense_44_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_44/MatMul/ReadVariableOp?
dense_44/MatMulMatMuldense_43/Relu:activations:0&dense_44/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_44/MatMul?
dense_44/BiasAdd/ReadVariableOpReadVariableOp(dense_44_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_44/BiasAdd/ReadVariableOp?
dense_44/BiasAddBiasAdddense_44/MatMul:product:0'dense_44/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_44/BiasAddt
dense_44/ReluReludense_44/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_44/Relu?
dense_45/MatMul/ReadVariableOpReadVariableOp'dense_45_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_45/MatMul/ReadVariableOp?
dense_45/MatMulMatMuldense_44/Relu:activations:0&dense_45/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_45/MatMul?
dense_45/BiasAdd/ReadVariableOpReadVariableOp(dense_45_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_45/BiasAdd/ReadVariableOp?
dense_45/BiasAddBiasAdddense_45/MatMul:product:0'dense_45/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_45/BiasAdd|
dense_45/SoftmaxSoftmaxdense_45/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_45/Softmax?
IdentityIdentitydense_45/Softmax:softmax:0 ^dense_41/BiasAdd/ReadVariableOp^dense_41/MatMul/ReadVariableOp ^dense_42/BiasAdd/ReadVariableOp^dense_42/MatMul/ReadVariableOp ^dense_43/BiasAdd/ReadVariableOp^dense_43/MatMul/ReadVariableOp ^dense_44/BiasAdd/ReadVariableOp^dense_44/MatMul/ReadVariableOp ^dense_45/BiasAdd/ReadVariableOp^dense_45/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????A: : : : : : : : : : 2B
dense_41/BiasAdd/ReadVariableOpdense_41/BiasAdd/ReadVariableOp2@
dense_41/MatMul/ReadVariableOpdense_41/MatMul/ReadVariableOp2B
dense_42/BiasAdd/ReadVariableOpdense_42/BiasAdd/ReadVariableOp2@
dense_42/MatMul/ReadVariableOpdense_42/MatMul/ReadVariableOp2B
dense_43/BiasAdd/ReadVariableOpdense_43/BiasAdd/ReadVariableOp2@
dense_43/MatMul/ReadVariableOpdense_43/MatMul/ReadVariableOp2B
dense_44/BiasAdd/ReadVariableOpdense_44/BiasAdd/ReadVariableOp2@
dense_44/MatMul/ReadVariableOpdense_44/MatMul/ReadVariableOp2B
dense_45/BiasAdd/ReadVariableOpdense_45/BiasAdd/ReadVariableOp2@
dense_45/MatMul/ReadVariableOpdense_45/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????A
 
_user_specified_nameinputs
?2
?
J__inference_sequential_9_layer_call_and_return_conditional_losses_22988645

inputs:
'dense_41_matmul_readvariableop_resource:	A?7
(dense_41_biasadd_readvariableop_resource:	?;
'dense_42_matmul_readvariableop_resource:
??7
(dense_42_biasadd_readvariableop_resource:	?;
'dense_43_matmul_readvariableop_resource:
??7
(dense_43_biasadd_readvariableop_resource:	?;
'dense_44_matmul_readvariableop_resource:
??7
(dense_44_biasadd_readvariableop_resource:	?:
'dense_45_matmul_readvariableop_resource:	?6
(dense_45_biasadd_readvariableop_resource:
identity??dense_41/BiasAdd/ReadVariableOp?dense_41/MatMul/ReadVariableOp?dense_42/BiasAdd/ReadVariableOp?dense_42/MatMul/ReadVariableOp?dense_43/BiasAdd/ReadVariableOp?dense_43/MatMul/ReadVariableOp?dense_44/BiasAdd/ReadVariableOp?dense_44/MatMul/ReadVariableOp?dense_45/BiasAdd/ReadVariableOp?dense_45/MatMul/ReadVariableOpo
dense_41/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????A2
dense_41/Cast?
dense_41/MatMul/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource*
_output_shapes
:	A?*
dtype02 
dense_41/MatMul/ReadVariableOp?
dense_41/MatMulMatMuldense_41/Cast:y:0&dense_41/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_41/MatMul?
dense_41/BiasAdd/ReadVariableOpReadVariableOp(dense_41_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_41/BiasAdd/ReadVariableOp?
dense_41/BiasAddBiasAdddense_41/MatMul:product:0'dense_41/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_41/BiasAddt
dense_41/ReluReludense_41/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_41/Relu?
dense_42/MatMul/ReadVariableOpReadVariableOp'dense_42_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_42/MatMul/ReadVariableOp?
dense_42/MatMulMatMuldense_41/Relu:activations:0&dense_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_42/MatMul?
dense_42/BiasAdd/ReadVariableOpReadVariableOp(dense_42_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_42/BiasAdd/ReadVariableOp?
dense_42/BiasAddBiasAdddense_42/MatMul:product:0'dense_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_42/BiasAddt
dense_42/ReluReludense_42/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_42/Relu?
dense_43/MatMul/ReadVariableOpReadVariableOp'dense_43_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_43/MatMul/ReadVariableOp?
dense_43/MatMulMatMuldense_42/Relu:activations:0&dense_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_43/MatMul?
dense_43/BiasAdd/ReadVariableOpReadVariableOp(dense_43_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_43/BiasAdd/ReadVariableOp?
dense_43/BiasAddBiasAdddense_43/MatMul:product:0'dense_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_43/BiasAddt
dense_43/ReluReludense_43/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_43/Relu?
dense_44/MatMul/ReadVariableOpReadVariableOp'dense_44_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_44/MatMul/ReadVariableOp?
dense_44/MatMulMatMuldense_43/Relu:activations:0&dense_44/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_44/MatMul?
dense_44/BiasAdd/ReadVariableOpReadVariableOp(dense_44_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_44/BiasAdd/ReadVariableOp?
dense_44/BiasAddBiasAdddense_44/MatMul:product:0'dense_44/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_44/BiasAddt
dense_44/ReluReludense_44/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_44/Relu?
dense_45/MatMul/ReadVariableOpReadVariableOp'dense_45_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_45/MatMul/ReadVariableOp?
dense_45/MatMulMatMuldense_44/Relu:activations:0&dense_45/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_45/MatMul?
dense_45/BiasAdd/ReadVariableOpReadVariableOp(dense_45_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_45/BiasAdd/ReadVariableOp?
dense_45/BiasAddBiasAdddense_45/MatMul:product:0'dense_45/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_45/BiasAdd|
dense_45/SoftmaxSoftmaxdense_45/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_45/Softmax?
IdentityIdentitydense_45/Softmax:softmax:0 ^dense_41/BiasAdd/ReadVariableOp^dense_41/MatMul/ReadVariableOp ^dense_42/BiasAdd/ReadVariableOp^dense_42/MatMul/ReadVariableOp ^dense_43/BiasAdd/ReadVariableOp^dense_43/MatMul/ReadVariableOp ^dense_44/BiasAdd/ReadVariableOp^dense_44/MatMul/ReadVariableOp ^dense_45/BiasAdd/ReadVariableOp^dense_45/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????A: : : : : : : : : : 2B
dense_41/BiasAdd/ReadVariableOpdense_41/BiasAdd/ReadVariableOp2@
dense_41/MatMul/ReadVariableOpdense_41/MatMul/ReadVariableOp2B
dense_42/BiasAdd/ReadVariableOpdense_42/BiasAdd/ReadVariableOp2@
dense_42/MatMul/ReadVariableOpdense_42/MatMul/ReadVariableOp2B
dense_43/BiasAdd/ReadVariableOpdense_43/BiasAdd/ReadVariableOp2@
dense_43/MatMul/ReadVariableOpdense_43/MatMul/ReadVariableOp2B
dense_44/BiasAdd/ReadVariableOpdense_44/BiasAdd/ReadVariableOp2@
dense_44/MatMul/ReadVariableOpdense_44/MatMul/ReadVariableOp2B
dense_45/BiasAdd/ReadVariableOpdense_45/BiasAdd/ReadVariableOp2@
dense_45/MatMul/ReadVariableOpdense_45/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????A
 
_user_specified_nameinputs
?

?
/__inference_sequential_9_layer_call_fn_22988468
dense_41_input
unknown:	A?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_41_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_9_layer_call_and_return_conditional_losses_229884202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????A: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????A
(
_user_specified_namedense_41_input
?

?
/__inference_sequential_9_layer_call_fn_22988314
dense_41_input
unknown:	A?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_41_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_9_layer_call_and_return_conditional_losses_229882912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????A: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????A
(
_user_specified_namedense_41_input
?

?
F__inference_dense_44_layer_call_and_return_conditional_losses_22988766

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
F__inference_dense_45_layer_call_and_return_conditional_losses_22988786

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
F__inference_dense_43_layer_call_and_return_conditional_losses_22988250

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_9_layer_call_and_return_conditional_losses_22988420

inputs$
dense_41_22988394:	A? 
dense_41_22988396:	?%
dense_42_22988399:
?? 
dense_42_22988401:	?%
dense_43_22988404:
?? 
dense_43_22988406:	?%
dense_44_22988409:
?? 
dense_44_22988411:	?$
dense_45_22988414:	?
dense_45_22988416:
identity?? dense_41/StatefulPartitionedCall? dense_42/StatefulPartitionedCall? dense_43/StatefulPartitionedCall? dense_44/StatefulPartitionedCall? dense_45/StatefulPartitionedCall?
 dense_41/StatefulPartitionedCallStatefulPartitionedCallinputsdense_41_22988394dense_41_22988396*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_41_layer_call_and_return_conditional_losses_229882162"
 dense_41/StatefulPartitionedCall?
 dense_42/StatefulPartitionedCallStatefulPartitionedCall)dense_41/StatefulPartitionedCall:output:0dense_42_22988399dense_42_22988401*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_42_layer_call_and_return_conditional_losses_229882332"
 dense_42/StatefulPartitionedCall?
 dense_43/StatefulPartitionedCallStatefulPartitionedCall)dense_42/StatefulPartitionedCall:output:0dense_43_22988404dense_43_22988406*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_43_layer_call_and_return_conditional_losses_229882502"
 dense_43/StatefulPartitionedCall?
 dense_44/StatefulPartitionedCallStatefulPartitionedCall)dense_43/StatefulPartitionedCall:output:0dense_44_22988409dense_44_22988411*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_44_layer_call_and_return_conditional_losses_229882672"
 dense_44/StatefulPartitionedCall?
 dense_45/StatefulPartitionedCallStatefulPartitionedCall)dense_44/StatefulPartitionedCall:output:0dense_45_22988414dense_45_22988416*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_45_layer_call_and_return_conditional_losses_229882842"
 dense_45/StatefulPartitionedCall?
IdentityIdentity)dense_45/StatefulPartitionedCall:output:0!^dense_41/StatefulPartitionedCall!^dense_42/StatefulPartitionedCall!^dense_43/StatefulPartitionedCall!^dense_44/StatefulPartitionedCall!^dense_45/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????A: : : : : : : : : : 2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2D
 dense_42/StatefulPartitionedCall dense_42/StatefulPartitionedCall2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall2D
 dense_44/StatefulPartitionedCall dense_44/StatefulPartitionedCall2D
 dense_45/StatefulPartitionedCall dense_45/StatefulPartitionedCall:O K
'
_output_shapes
:?????????A
 
_user_specified_nameinputs
?@
?	
#__inference__wrapped_model_22988197
dense_41_inputG
4sequential_9_dense_41_matmul_readvariableop_resource:	A?D
5sequential_9_dense_41_biasadd_readvariableop_resource:	?H
4sequential_9_dense_42_matmul_readvariableop_resource:
??D
5sequential_9_dense_42_biasadd_readvariableop_resource:	?H
4sequential_9_dense_43_matmul_readvariableop_resource:
??D
5sequential_9_dense_43_biasadd_readvariableop_resource:	?H
4sequential_9_dense_44_matmul_readvariableop_resource:
??D
5sequential_9_dense_44_biasadd_readvariableop_resource:	?G
4sequential_9_dense_45_matmul_readvariableop_resource:	?C
5sequential_9_dense_45_biasadd_readvariableop_resource:
identity??,sequential_9/dense_41/BiasAdd/ReadVariableOp?+sequential_9/dense_41/MatMul/ReadVariableOp?,sequential_9/dense_42/BiasAdd/ReadVariableOp?+sequential_9/dense_42/MatMul/ReadVariableOp?,sequential_9/dense_43/BiasAdd/ReadVariableOp?+sequential_9/dense_43/MatMul/ReadVariableOp?,sequential_9/dense_44/BiasAdd/ReadVariableOp?+sequential_9/dense_44/MatMul/ReadVariableOp?,sequential_9/dense_45/BiasAdd/ReadVariableOp?+sequential_9/dense_45/MatMul/ReadVariableOp?
sequential_9/dense_41/CastCastdense_41_input*

DstT0*

SrcT0*'
_output_shapes
:?????????A2
sequential_9/dense_41/Cast?
+sequential_9/dense_41/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_41_matmul_readvariableop_resource*
_output_shapes
:	A?*
dtype02-
+sequential_9/dense_41/MatMul/ReadVariableOp?
sequential_9/dense_41/MatMulMatMulsequential_9/dense_41/Cast:y:03sequential_9/dense_41/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_9/dense_41/MatMul?
,sequential_9/dense_41/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_41_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,sequential_9/dense_41/BiasAdd/ReadVariableOp?
sequential_9/dense_41/BiasAddBiasAdd&sequential_9/dense_41/MatMul:product:04sequential_9/dense_41/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_9/dense_41/BiasAdd?
sequential_9/dense_41/ReluRelu&sequential_9/dense_41/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_9/dense_41/Relu?
+sequential_9/dense_42/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_42_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+sequential_9/dense_42/MatMul/ReadVariableOp?
sequential_9/dense_42/MatMulMatMul(sequential_9/dense_41/Relu:activations:03sequential_9/dense_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_9/dense_42/MatMul?
,sequential_9/dense_42/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_42_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,sequential_9/dense_42/BiasAdd/ReadVariableOp?
sequential_9/dense_42/BiasAddBiasAdd&sequential_9/dense_42/MatMul:product:04sequential_9/dense_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_9/dense_42/BiasAdd?
sequential_9/dense_42/ReluRelu&sequential_9/dense_42/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_9/dense_42/Relu?
+sequential_9/dense_43/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_43_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+sequential_9/dense_43/MatMul/ReadVariableOp?
sequential_9/dense_43/MatMulMatMul(sequential_9/dense_42/Relu:activations:03sequential_9/dense_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_9/dense_43/MatMul?
,sequential_9/dense_43/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_43_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,sequential_9/dense_43/BiasAdd/ReadVariableOp?
sequential_9/dense_43/BiasAddBiasAdd&sequential_9/dense_43/MatMul:product:04sequential_9/dense_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_9/dense_43/BiasAdd?
sequential_9/dense_43/ReluRelu&sequential_9/dense_43/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_9/dense_43/Relu?
+sequential_9/dense_44/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_44_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+sequential_9/dense_44/MatMul/ReadVariableOp?
sequential_9/dense_44/MatMulMatMul(sequential_9/dense_43/Relu:activations:03sequential_9/dense_44/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_9/dense_44/MatMul?
,sequential_9/dense_44/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_44_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,sequential_9/dense_44/BiasAdd/ReadVariableOp?
sequential_9/dense_44/BiasAddBiasAdd&sequential_9/dense_44/MatMul:product:04sequential_9/dense_44/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_9/dense_44/BiasAdd?
sequential_9/dense_44/ReluRelu&sequential_9/dense_44/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_9/dense_44/Relu?
+sequential_9/dense_45/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_45_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02-
+sequential_9/dense_45/MatMul/ReadVariableOp?
sequential_9/dense_45/MatMulMatMul(sequential_9/dense_44/Relu:activations:03sequential_9/dense_45/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_9/dense_45/MatMul?
,sequential_9/dense_45/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_45_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_9/dense_45/BiasAdd/ReadVariableOp?
sequential_9/dense_45/BiasAddBiasAdd&sequential_9/dense_45/MatMul:product:04sequential_9/dense_45/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_9/dense_45/BiasAdd?
sequential_9/dense_45/SoftmaxSoftmax&sequential_9/dense_45/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_9/dense_45/Softmax?
IdentityIdentity'sequential_9/dense_45/Softmax:softmax:0-^sequential_9/dense_41/BiasAdd/ReadVariableOp,^sequential_9/dense_41/MatMul/ReadVariableOp-^sequential_9/dense_42/BiasAdd/ReadVariableOp,^sequential_9/dense_42/MatMul/ReadVariableOp-^sequential_9/dense_43/BiasAdd/ReadVariableOp,^sequential_9/dense_43/MatMul/ReadVariableOp-^sequential_9/dense_44/BiasAdd/ReadVariableOp,^sequential_9/dense_44/MatMul/ReadVariableOp-^sequential_9/dense_45/BiasAdd/ReadVariableOp,^sequential_9/dense_45/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????A: : : : : : : : : : 2\
,sequential_9/dense_41/BiasAdd/ReadVariableOp,sequential_9/dense_41/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_41/MatMul/ReadVariableOp+sequential_9/dense_41/MatMul/ReadVariableOp2\
,sequential_9/dense_42/BiasAdd/ReadVariableOp,sequential_9/dense_42/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_42/MatMul/ReadVariableOp+sequential_9/dense_42/MatMul/ReadVariableOp2\
,sequential_9/dense_43/BiasAdd/ReadVariableOp,sequential_9/dense_43/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_43/MatMul/ReadVariableOp+sequential_9/dense_43/MatMul/ReadVariableOp2\
,sequential_9/dense_44/BiasAdd/ReadVariableOp,sequential_9/dense_44/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_44/MatMul/ReadVariableOp+sequential_9/dense_44/MatMul/ReadVariableOp2\
,sequential_9/dense_45/BiasAdd/ReadVariableOp,sequential_9/dense_45/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_45/MatMul/ReadVariableOp+sequential_9/dense_45/MatMul/ReadVariableOp:W S
'
_output_shapes
:?????????A
(
_user_specified_namedense_41_input
?

?
F__inference_dense_42_layer_call_and_return_conditional_losses_22988233

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_9_layer_call_and_return_conditional_losses_22988497
dense_41_input$
dense_41_22988471:	A? 
dense_41_22988473:	?%
dense_42_22988476:
?? 
dense_42_22988478:	?%
dense_43_22988481:
?? 
dense_43_22988483:	?%
dense_44_22988486:
?? 
dense_44_22988488:	?$
dense_45_22988491:	?
dense_45_22988493:
identity?? dense_41/StatefulPartitionedCall? dense_42/StatefulPartitionedCall? dense_43/StatefulPartitionedCall? dense_44/StatefulPartitionedCall? dense_45/StatefulPartitionedCall?
 dense_41/StatefulPartitionedCallStatefulPartitionedCalldense_41_inputdense_41_22988471dense_41_22988473*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_41_layer_call_and_return_conditional_losses_229882162"
 dense_41/StatefulPartitionedCall?
 dense_42/StatefulPartitionedCallStatefulPartitionedCall)dense_41/StatefulPartitionedCall:output:0dense_42_22988476dense_42_22988478*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_42_layer_call_and_return_conditional_losses_229882332"
 dense_42/StatefulPartitionedCall?
 dense_43/StatefulPartitionedCallStatefulPartitionedCall)dense_42/StatefulPartitionedCall:output:0dense_43_22988481dense_43_22988483*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_43_layer_call_and_return_conditional_losses_229882502"
 dense_43/StatefulPartitionedCall?
 dense_44/StatefulPartitionedCallStatefulPartitionedCall)dense_43/StatefulPartitionedCall:output:0dense_44_22988486dense_44_22988488*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_44_layer_call_and_return_conditional_losses_229882672"
 dense_44/StatefulPartitionedCall?
 dense_45/StatefulPartitionedCallStatefulPartitionedCall)dense_44/StatefulPartitionedCall:output:0dense_45_22988491dense_45_22988493*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_45_layer_call_and_return_conditional_losses_229882842"
 dense_45/StatefulPartitionedCall?
IdentityIdentity)dense_45/StatefulPartitionedCall:output:0!^dense_41/StatefulPartitionedCall!^dense_42/StatefulPartitionedCall!^dense_43/StatefulPartitionedCall!^dense_44/StatefulPartitionedCall!^dense_45/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????A: : : : : : : : : : 2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2D
 dense_42/StatefulPartitionedCall dense_42/StatefulPartitionedCall2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall2D
 dense_44/StatefulPartitionedCall dense_44/StatefulPartitionedCall2D
 dense_45/StatefulPartitionedCall dense_45/StatefulPartitionedCall:W S
'
_output_shapes
:?????????A
(
_user_specified_namedense_41_input
?

?
F__inference_dense_43_layer_call_and_return_conditional_losses_22988746

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_dense_41_layer_call_and_return_conditional_losses_22988706

inputs1
matmul_readvariableop_resource:	A?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????A2
Cast?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	A?*
dtype02
MatMul/ReadVariableOpv
MatMulMatMulCast:y:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????A
 
_user_specified_nameinputs
?

?
/__inference_sequential_9_layer_call_fn_22988580

inputs
unknown:	A?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_9_layer_call_and_return_conditional_losses_229882912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????A: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????A
 
_user_specified_nameinputs
?

?
F__inference_dense_45_layer_call_and_return_conditional_losses_22988284

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_9_layer_call_and_return_conditional_losses_22988526
dense_41_input$
dense_41_22988500:	A? 
dense_41_22988502:	?%
dense_42_22988505:
?? 
dense_42_22988507:	?%
dense_43_22988510:
?? 
dense_43_22988512:	?%
dense_44_22988515:
?? 
dense_44_22988517:	?$
dense_45_22988520:	?
dense_45_22988522:
identity?? dense_41/StatefulPartitionedCall? dense_42/StatefulPartitionedCall? dense_43/StatefulPartitionedCall? dense_44/StatefulPartitionedCall? dense_45/StatefulPartitionedCall?
 dense_41/StatefulPartitionedCallStatefulPartitionedCalldense_41_inputdense_41_22988500dense_41_22988502*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_41_layer_call_and_return_conditional_losses_229882162"
 dense_41/StatefulPartitionedCall?
 dense_42/StatefulPartitionedCallStatefulPartitionedCall)dense_41/StatefulPartitionedCall:output:0dense_42_22988505dense_42_22988507*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_42_layer_call_and_return_conditional_losses_229882332"
 dense_42/StatefulPartitionedCall?
 dense_43/StatefulPartitionedCallStatefulPartitionedCall)dense_42/StatefulPartitionedCall:output:0dense_43_22988510dense_43_22988512*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_43_layer_call_and_return_conditional_losses_229882502"
 dense_43/StatefulPartitionedCall?
 dense_44/StatefulPartitionedCallStatefulPartitionedCall)dense_43/StatefulPartitionedCall:output:0dense_44_22988515dense_44_22988517*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_44_layer_call_and_return_conditional_losses_229882672"
 dense_44/StatefulPartitionedCall?
 dense_45/StatefulPartitionedCallStatefulPartitionedCall)dense_44/StatefulPartitionedCall:output:0dense_45_22988520dense_45_22988522*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_45_layer_call_and_return_conditional_losses_229882842"
 dense_45/StatefulPartitionedCall?
IdentityIdentity)dense_45/StatefulPartitionedCall:output:0!^dense_41/StatefulPartitionedCall!^dense_42/StatefulPartitionedCall!^dense_43/StatefulPartitionedCall!^dense_44/StatefulPartitionedCall!^dense_45/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????A: : : : : : : : : : 2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2D
 dense_42/StatefulPartitionedCall dense_42/StatefulPartitionedCall2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall2D
 dense_44/StatefulPartitionedCall dense_44/StatefulPartitionedCall2D
 dense_45/StatefulPartitionedCall dense_45/StatefulPartitionedCall:W S
'
_output_shapes
:?????????A
(
_user_specified_namedense_41_input
?w
?
$__inference__traced_restore_22988981
file_prefix3
 assignvariableop_dense_41_kernel:	A?/
 assignvariableop_1_dense_41_bias:	?6
"assignvariableop_2_dense_42_kernel:
??/
 assignvariableop_3_dense_42_bias:	?6
"assignvariableop_4_dense_43_kernel:
??/
 assignvariableop_5_dense_43_bias:	?6
"assignvariableop_6_dense_44_kernel:
??/
 assignvariableop_7_dense_44_bias:	?5
"assignvariableop_8_dense_45_kernel:	?.
 assignvariableop_9_dense_45_bias:*
 assignvariableop_10_adagrad_iter:	 +
!assignvariableop_11_adagrad_decay: 3
)assignvariableop_12_adagrad_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: J
7assignvariableop_17_adagrad_dense_41_kernel_accumulator:	A?D
5assignvariableop_18_adagrad_dense_41_bias_accumulator:	?K
7assignvariableop_19_adagrad_dense_42_kernel_accumulator:
??D
5assignvariableop_20_adagrad_dense_42_bias_accumulator:	?K
7assignvariableop_21_adagrad_dense_43_kernel_accumulator:
??D
5assignvariableop_22_adagrad_dense_43_bias_accumulator:	?K
7assignvariableop_23_adagrad_dense_44_kernel_accumulator:
??D
5assignvariableop_24_adagrad_dense_44_bias_accumulator:	?J
7assignvariableop_25_adagrad_dense_45_kernel_accumulator:	?C
5assignvariableop_26_adagrad_dense_45_bias_accumulator:
identity_28??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_41_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_41_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_42_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_42_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_43_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_43_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_44_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_44_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_45_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_45_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp assignvariableop_10_adagrad_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_adagrad_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp)assignvariableop_12_adagrad_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp7assignvariableop_17_adagrad_dense_41_kernel_accumulatorIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp5assignvariableop_18_adagrad_dense_41_bias_accumulatorIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp7assignvariableop_19_adagrad_dense_42_kernel_accumulatorIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp5assignvariableop_20_adagrad_dense_42_bias_accumulatorIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp7assignvariableop_21_adagrad_dense_43_kernel_accumulatorIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp5assignvariableop_22_adagrad_dense_43_bias_accumulatorIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp7assignvariableop_23_adagrad_dense_44_kernel_accumulatorIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp5assignvariableop_24_adagrad_dense_44_bias_accumulatorIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp7assignvariableop_25_adagrad_dense_45_kernel_accumulatorIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp5assignvariableop_26_adagrad_dense_45_bias_accumulatorIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_269
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_27?
Identity_28IdentityIdentity_27:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_28"#
identity_28Identity_28:output:0*K
_input_shapes:
8: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
I
dense_41_input7
 serving_default_dense_41_input:0?????????A<
dense_450
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?5
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
`__call__
a_default_save_signature
*b&call_and_return_all_conditional_losses"?2
_tf_keras_sequential?2{"name": "sequential_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 65]}, "dtype": "uint16", "sparse": false, "ragged": false, "name": "dense_41_input"}}, {"class_name": "Dense", "config": {"name": "dense_41", "trainable": true, "dtype": "float32", "units": 195, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_42", "trainable": true, "dtype": "float32", "units": 195, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_43", "trainable": true, "dtype": "float32", "units": 195, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_44", "trainable": true, "dtype": "float32", "units": 195, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_45", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 65}}, "shared_object_id": 17}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 65]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 65]}, "uint16", "dense_41_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 65]}, "dtype": "uint16", "sparse": false, "ragged": false, "name": "dense_41_input"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_41", "trainable": true, "dtype": "float32", "units": 195, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "Dense", "config": {"name": "dense_42", "trainable": true, "dtype": "float32", "units": 195, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6}, {"class_name": "Dense", "config": {"name": "dense_43", "trainable": true, "dtype": "float32", "units": 195, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 9}, {"class_name": "Dense", "config": {"name": "dense_44", "trainable": true, "dtype": "float32", "units": 195, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, {"class_name": "Dense", "config": {"name": "dense_45", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 15}]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": false}, "shared_object_id": 18}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 19}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adagrad", "config": {"name": "Adagrad", "learning_rate": 0.0010000000474974513, "decay": 0.0, "initial_accumulator_value": 0.1, "epsilon": 1e-07}}}}
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
c__call__
*d&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_41", "trainable": true, "dtype": "float32", "units": 195, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 65}}, "shared_object_id": 17}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 65]}}
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
e__call__
*f&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_42", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_42", "trainable": true, "dtype": "float32", "units": 195, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 195}}, "shared_object_id": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 195]}}
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
g__call__
*h&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_43", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_43", "trainable": true, "dtype": "float32", "units": 195, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 195}}, "shared_object_id": 21}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 195]}}
?

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
i__call__
*j&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_44", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_44", "trainable": true, "dtype": "float32", "units": 195, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 195}}, "shared_object_id": 22}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 195]}}
?

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
k__call__
*l&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_45", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_45", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 195}}, "shared_object_id": 23}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 195]}}
?
*iter
	+decay
,learning_rateaccumulatorVaccumulatorWaccumulatorXaccumulatorYaccumulatorZaccumulator[accumulator\accumulator]$accumulator^%accumulator_"
	optimizer
f
0
1
2
3
4
5
6
7
$8
%9"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
$8
%9"
trackable_list_wrapper
 "
trackable_list_wrapper
?
-metrics
	variables
.layer_regularization_losses
/non_trainable_variables
trainable_variables
0layer_metrics

1layers
	regularization_losses
`__call__
a_default_save_signature
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
,
mserving_default"
signature_map
": 	A?2dense_41/kernel
:?2dense_41/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
2metrics
	variables
3layer_regularization_losses
4non_trainable_variables
trainable_variables
5layer_metrics

6layers
regularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_42/kernel
:?2dense_42/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
7metrics
	variables
8layer_regularization_losses
9non_trainable_variables
trainable_variables
:layer_metrics

;layers
regularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_43/kernel
:?2dense_43/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
<metrics
	variables
=layer_regularization_losses
>non_trainable_variables
trainable_variables
?layer_metrics

@layers
regularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_44/kernel
:?2dense_44/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ametrics
 	variables
Blayer_regularization_losses
Cnon_trainable_variables
!trainable_variables
Dlayer_metrics

Elayers
"regularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_45/kernel
:2dense_45/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Fmetrics
&	variables
Glayer_regularization_losses
Hnon_trainable_variables
'trainable_variables
Ilayer_metrics

Jlayers
(regularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
:	 (2Adagrad/iter
: (2Adagrad/decay
: (2Adagrad/learning_rate
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
C
0
1
2
3
4"
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
?
	Mtotal
	Ncount
O	variables
P	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 24}
?
	Qtotal
	Rcount
S
_fn_kwargs
T	variables
U	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 19}
:  (2total
:  (2count
.
M0
N1"
trackable_list_wrapper
-
O	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
Q0
R1"
trackable_list_wrapper
-
T	variables"
_generic_user_object
4:2	A?2#Adagrad/dense_41/kernel/accumulator
.:,?2!Adagrad/dense_41/bias/accumulator
5:3
??2#Adagrad/dense_42/kernel/accumulator
.:,?2!Adagrad/dense_42/bias/accumulator
5:3
??2#Adagrad/dense_43/kernel/accumulator
.:,?2!Adagrad/dense_43/bias/accumulator
5:3
??2#Adagrad/dense_44/kernel/accumulator
.:,?2!Adagrad/dense_44/bias/accumulator
4:2	?2#Adagrad/dense_45/kernel/accumulator
-:+2!Adagrad/dense_45/bias/accumulator
?2?
/__inference_sequential_9_layer_call_fn_22988314
/__inference_sequential_9_layer_call_fn_22988580
/__inference_sequential_9_layer_call_fn_22988605
/__inference_sequential_9_layer_call_fn_22988468?
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
?2?
#__inference__wrapped_model_22988197?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *-?*
(?%
dense_41_input?????????A
?2?
J__inference_sequential_9_layer_call_and_return_conditional_losses_22988645
J__inference_sequential_9_layer_call_and_return_conditional_losses_22988685
J__inference_sequential_9_layer_call_and_return_conditional_losses_22988497
J__inference_sequential_9_layer_call_and_return_conditional_losses_22988526?
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
?2?
+__inference_dense_41_layer_call_fn_22988694?
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
F__inference_dense_41_layer_call_and_return_conditional_losses_22988706?
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
+__inference_dense_42_layer_call_fn_22988715?
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
F__inference_dense_42_layer_call_and_return_conditional_losses_22988726?
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
+__inference_dense_43_layer_call_fn_22988735?
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
F__inference_dense_43_layer_call_and_return_conditional_losses_22988746?
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
+__inference_dense_44_layer_call_fn_22988755?
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
F__inference_dense_44_layer_call_and_return_conditional_losses_22988766?
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
+__inference_dense_45_layer_call_fn_22988775?
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
F__inference_dense_45_layer_call_and_return_conditional_losses_22988786?
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
?B?
&__inference_signature_wrapper_22988555dense_41_input"?
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
 ?
#__inference__wrapped_model_22988197z
$%7?4
-?*
(?%
dense_41_input?????????A
? "3?0
.
dense_45"?
dense_45??????????
F__inference_dense_41_layer_call_and_return_conditional_losses_22988706]/?,
%?"
 ?
inputs?????????A
? "&?#
?
0??????????
? 
+__inference_dense_41_layer_call_fn_22988694P/?,
%?"
 ?
inputs?????????A
? "????????????
F__inference_dense_42_layer_call_and_return_conditional_losses_22988726^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_42_layer_call_fn_22988715Q0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_43_layer_call_and_return_conditional_losses_22988746^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_43_layer_call_fn_22988735Q0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_44_layer_call_and_return_conditional_losses_22988766^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_44_layer_call_fn_22988755Q0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_45_layer_call_and_return_conditional_losses_22988786]$%0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? 
+__inference_dense_45_layer_call_fn_22988775P$%0?-
&?#
!?
inputs??????????
? "???????????
J__inference_sequential_9_layer_call_and_return_conditional_losses_22988497t
$%??<
5?2
(?%
dense_41_input?????????A
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_9_layer_call_and_return_conditional_losses_22988526t
$%??<
5?2
(?%
dense_41_input?????????A
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_9_layer_call_and_return_conditional_losses_22988645l
$%7?4
-?*
 ?
inputs?????????A
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_9_layer_call_and_return_conditional_losses_22988685l
$%7?4
-?*
 ?
inputs?????????A
p

 
? "%?"
?
0?????????
? ?
/__inference_sequential_9_layer_call_fn_22988314g
$%??<
5?2
(?%
dense_41_input?????????A
p 

 
? "???????????
/__inference_sequential_9_layer_call_fn_22988468g
$%??<
5?2
(?%
dense_41_input?????????A
p

 
? "???????????
/__inference_sequential_9_layer_call_fn_22988580_
$%7?4
-?*
 ?
inputs?????????A
p 

 
? "???????????
/__inference_sequential_9_layer_call_fn_22988605_
$%7?4
-?*
 ?
inputs?????????A
p

 
? "???????????
&__inference_signature_wrapper_22988555?
$%I?F
? 
??<
:
dense_41_input(?%
dense_41_input?????????A"3?0
.
dense_45"?
dense_45?????????