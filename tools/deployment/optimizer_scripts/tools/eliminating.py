import collections
import struct
import onnx
import numpy as np
from . import other
from . import helper
from . import modhelper
from .general_graph import Graph


def eliminate_Identify_and_Dropout(g):
    """
    Eliminate Identify layers

    :param g: the onnx graph
    """
    node_to_remove = []
    for node in g.node:
        if node.op_type != "Identity" and node.op_type != "Dropout":
            continue
        # If this node is the last, leave it to `eliminate_useless_last node`
        if helper.find_output_by_name(g, node.output[0]) is not None:
            continue
        # Replace the parents in all the following nodes
        following_nodes = helper.find_following_nodes_by_input_value_name(
            g, node.output[0]
        )
        for following_node in following_nodes:
            modhelper.replace_node_input(
                following_node, node.output[0], node.input[0]
            )
        # Delete value info
        value_between = helper.find_value_by_name(g, node.output[0])
        try:
            g.value_info.remove(value_between)
        except Exception:
            print("No value info to delete while eliminating identity layers.")
        # Node is waiting for elimination
        node_to_remove.append(node)
    for node in node_to_remove:
        g.node.remove(node)


# Remove last useless nodes
def remove_useless_last_nodes(g):
    """Remove useless nodes from the tail of the graph"""
    USELESS = [
        "Reshape",
        "Identity",
        "Transpose",
        "Flatten",
        "Dropout",
        "Mystery",
        "Constant",
        "Squeeze",
        "Unsqueeze",
        "Softmax",
    ]
    graph = Graph(g)
    todo = collections.deque()
    for node in graph.output_nodes:
        if len(node.children) == 0:
            todo.append(node)
    node_to_remove = []
    while todo:
        # BFS find nodes to remove
        cur_node = todo.popleft()
        if cur_node.proto is None:
            continue
        if cur_node.proto.op_type not in USELESS:
            continue
        # Find the output
        cur_node_output = helper.find_output_by_name(
            g, cur_node.proto.output[0]
        )
        for cur_input in cur_node.parents:
            cur_input.children.remove(cur_node)
            if len(cur_input.children) == 0:
                todo.append(cur_input)
            if cur_node_output is not None:
                cur_input_output = helper.find_value_by_name(
                    g, cur_input.proto.output[0]
                )
                cur_input_output_in_output = helper.find_output_by_name(
                    g, cur_input.proto.output[0]
                )
                if (
                    cur_input_output is not None
                    and cur_input_output_in_output is None
                ):
                    g.output.extend([cur_input_output])
        node_to_remove.append(cur_node.proto)
        try:
            g.value_info.remove(
                helper.find_value_by_name(g, cur_node.proto.output[0])
            )
        except ValueError:
            pass
        if cur_node_output is not None:
            g.output.remove(cur_node_output)
        cur_node.proto = None
        cur_node.parents.clear()
    for node in node_to_remove:
        g.node.remove(node)


######################################
#  TF only optimization passes       #
######################################


def eliminate_shape_changing_after_input(g):
    """
    Eliminate the Reshape node after input and reshape the input

    :param g: the onnx graph
    """
    node_to_remove = []
    REMOVE_LIST = [
        "Reshape",
        "Transpose",
        "Flatten",
        "Dropout",
        "Squeeze",
        "Unsqueeze",
    ]
    for node in g.node:
        # Find an input and the shape node
        if node.op_type not in REMOVE_LIST:
            continue
        old_input = helper.find_input_by_name(g, node.input[0])
        if old_input is None:
            continue
        # If the input is used by multiple nodes, skip.
        counter = 0
        for tnode in g.node:
            if old_input.name in tnode.input:
                counter += 1
        if counter > 1:
            continue
        # Remove Weight if any.
        output_val_info = helper.find_value_by_name(g, node.output[0])

        if node.op_type == "Reshape":
            shape_node = helper.find_node_by_output_name(g, node.input[1])
            if shape_node.op_type != "Constant":
                continue

            # manuelly set the input shape
            shape_info = helper.find_value_by_name(g, shape_node.output[0])
            old_size, old_shape = helper.find_size_shape_from_value(shape_info)

            _, new_shape = helper.constant_to_list(shape_node)
            for i in range(len(new_shape)):
                if new_shape[i] == -1:
                    dim = int(old_size // np.prod(new_shape) * (-1))
                    new_shape[i] = dim
            new_input = onnx.helper.make_tensor_value_info(
                output_val_info.name,
                output_val_info.type.tensor_type.elem_type,
                new_shape,
            )

            node_to_remove.append(node)

            shape_outputs = helper.find_nodes_by_input_name(
                g, shape_node.output[0]
            )
            if len(shape_outputs) == 1:
                node_to_remove.append(shape_node)
                g.value_info.remove(
                    helper.find_value_by_name(g, shape_node.output[0])
                )

            g.input.remove(old_input)
            g.input.extend([new_input])
            g.value_info.remove(output_val_info)
        elif node.op_type == "Transpose":
            permutation = list(node.attribute[0].ints)
            pre_shape = helper.get_shape_from_value_info(old_input)
            new_shape = [pre_shape[i] for i in permutation]

            new_input = onnx.helper.make_tensor_value_info(
                output_val_info.name,
                output_val_info.type.tensor_type.elem_type,
                new_shape,
            )

            node_to_remove.append(node)

            g.input.remove(old_input)
            g.input.extend([new_input])
            g.value_info.remove(output_val_info)
        elif node.op_type == "Flatten":
            axis = node.attribute[0].int
            pre_shape = helper.get_shape_from_value_info(old_input)
            dim_1, dim_2 = 1, 1
            if axis == 0:
                dim_1 = 1
                dim_2 = np.prod(pre_shape)
            else:
                dim_1 = np.prod(pre_shape[:axis]).astype(int)
                dim_2 = np.prod(pre_shape[axis:]).astype(int)
            new_shape = [dim_1, dim_2]

            new_input = onnx.helper.make_tensor_value_info(
                output_val_info.name,
                output_val_info.type.tensor_type.elem_type,
                new_shape,
            )

            node_to_remove.append(node)

            g.input.remove(old_input)
            g.input.extend([new_input])
            g.value_info.remove(output_val_info)
        elif node.op_type == "Dropout":
            g.input.remove(old_input)
            g.input.extend([output_val_info])
            g.value_info.remove(output_val_info)

            node_to_remove.append(node)
        elif node.op_type == "Squeeze":
            axis = list(node.attribute[0].ints)
            pre_shape = helper.get_shape_from_value_info(old_input)
            for pos in sorted(axis)[::-1]:
                if pre_shape[pos] != 1:
                    raise RuntimeError("invalid axis for squeeze")
                else:
                    pre_shape.pop(pos)
            new_shape = pre_shape

            new_input = onnx.helper.make_tensor_value_info(
                output_val_info.name,
                output_val_info.type.tensor_type.elem_type,
                new_shape,
            )

            node_to_remove.append(node)

            g.input.remove(old_input)
            g.input.extend([new_input])
            g.value_info.remove(output_val_info)
        elif node.op_type == "Unsqueeze":
            axis = list(node.attribute[0].ints)
            pre_shape = helper.get_shape_from_value_info(old_input)
            new_shape = pre_shape
            for pos in axis:
                new_shape.insert(pos, 1)
            new_input = onnx.helper.make_tensor_value_info(
                output_val_info.name,
                output_val_info.type.tensor_type.elem_type,
                new_shape,
            )
            node_to_remove.append(node)

            g.input.remove(old_input)
            g.input.extend([new_input])
            g.value_info.remove(output_val_info)
        else:
            pass

    for node in node_to_remove:
        g.node.remove(node)

    other.topological_sort(g)


def eliminate_Reshape_Cast(g):
    """Eliminate the cast layer for shape of Reshape layer

    :param g: the onnx graph
    """
    # Find all reshape layers
    for node in g.node:
        if node.op_type != "Reshape":
            continue
        prev_node = helper.find_node_by_output_name(g, node.input[1])
        if prev_node.op_type != "Cast":
            continue
        reshape_node = node
        cast_node = prev_node
        weight_node = helper.find_node_by_output_name(g, cast_node.input[0])
        if weight_node is None:
            raise RuntimeError("Unexpected None before Cast-Reshape.")
        weight_node.attribute[0].t.data_type = 7
        if weight_node.attribute[0].t.raw_data:
            raw_data = weight_node.attribute[0].t.raw_data
            int_data = [i[0] for i in struct.iter_unpack("i", raw_data)]
            raw_data = struct.pack("q" * len(int_data), *int_data)
        elif (
            len(weight_node.attribute[0].t.int64_data) > 0
            or len(weight_node.attribute[0].t.int32_data) > 0
        ):
            # It's already int. Do nothing
            pass
        else:
            raise NotImplementedError()
        # Change Value info
        origin_weight_out = helper.find_value_by_name(g, weight_node.output[0])
        weight_node.output.pop()
        weight_node.output.extend([reshape_node.input[1]])
        # Delete
        g.value_info.remove(origin_weight_out)
        g.node.remove(cast_node)


def eliminate_Cast_after_input(g):
    """Eliminate the cast layer right after the input

    :param g: the onnx graph
    """
    node_to_remove = []
    for node in g.node:
        if node.op_type != "Cast":
            continue
        old_input = helper.find_input_by_name(g, node.input[0])
        if old_input is None:
            continue
        next_val_info = helper.find_value_by_name(g, node.output[0])
        shape = helper.get_shape_from_value_info(next_val_info)
        new_val_info = onnx.helper.make_tensor_value_info(
            next_val_info.name, node.attribute[0].i, shape
        )
        # Delete old value_info
        g.input.remove(old_input)
        g.value_info.remove(next_val_info)
        # Append nodes to node_to_remove
        node_to_remove.append(node)
        # Add new input
        g.input.extend([new_val_info])
    for node in node_to_remove:
        g.node.remove(node)


def eliminate_consecutive_Cast(g):
    """If two cast is next to each other, remove the first cast

    :param g: the onnx graph
    """
    node_to_remove = []
    for node in g.node:
        if node.op_type != "Cast":
            continue
        first_node = helper.find_node_by_output_name(g, node.input[0])
        if first_node is None or first_node.op_type != "Cast":
            continue
        # Here we have two consecutive Cast Node
        # Reset the input of the later node
        node.input[0] = first_node.input[0]
        # Remove the first node and its output value info
        node_to_remove.append(first_node)
        first_output = helper.find_value_by_name(g, first_node.output[0])
        g.value_info.remove(first_output)
    for node in node_to_remove:
        g.node.remove(node)


def eliminate_Squeeze_before_Reshape(g):
    """If Squeeze and Reshape is next to each other, remove the first node

    :param g: the onnx graph
    """
    node_to_remove = []
    for node in g.node:
        if node.op_type != "Reshape":
            continue
        first_node = helper.find_node_by_output_name(g, node.input[0])
        if not first_node:
            continue
        if first_node.op_type != "Squeeze":
            continue
        # Here we have two consecutive Cast Node
        # Reset the input of the later node
        node.input[0] = first_node.input[0]
        # Remove the first node and its output value info
        node_to_remove.append(first_node)
        first_output = helper.find_value_by_name(g, first_node.output[0])
        g.value_info.remove(first_output)
    for node in node_to_remove:
        g.node.remove(node)


def eliminate_no_children_input(g):
    """Eliminate inputs with no children at all."""
    # Create a set of input names
    input_names = set([i.name for i in g.input])
    # If a name is used in any node, remove this name from the set.
    for n in g.node:
        for i in n.input:
            input_names.discard(i)
    # Remove the inputs with the left names.
    for i in input_names:
        info = helper.find_input_by_name(g, i)
        g.input.remove(info)


def eliminate_consecutive_reshape(g):
    """Replace consecutive reshape nodes by a single node."""
    node_to_del = []
    for node in g.node:
        if node.op_type != "Reshape":
            continue
        pre_data_node = helper.find_node_by_output_name(g, node.input[0])
        pre_shape_node = helper.find_node_by_output_name(g, node.input[1])
        if not pre_data_node or not pre_shape_node:
            continue
        if pre_shape_node.op_type != "Constant":
            continue
        if pre_data_node.op_type != "Reshape":
            continue

        pre_pre_shape_node = helper.find_node_by_output_name(
            g, pre_data_node.input[1]
        )
        if pre_pre_shape_node.op_type != "Constant":
            continue

        new_reshape_node = onnx.helper.make_node(
            "Reshape",
            [pre_data_node.input[0], node.input[1]],
            [node.output[0]],
            name=node.output[0],
        )

        g.node.extend([new_reshape_node])
        node_to_del.append(node)
        node_to_del.append(pre_data_node)
        node_to_del.append(pre_pre_shape_node)

        val_info_to_del1 = helper.find_value_by_name(g, node.input[0])
        val_info_to_del2 = helper.find_value_by_name(g, pre_data_node.input[1])
        g.value_info.remove(val_info_to_del1)
        g.value_info.remove(val_info_to_del2)

    while node_to_del:
        node = node_to_del.pop()
        g.node.remove(node)


def eliminate_single_input_Concat(g):
    """
    Eliminate single input Concat layers

    :param g: the onnx graph
    """
    node_to_remove = []
    for node in g.node:
        if node.op_type != "Concat":
            continue
        # If this node has more than 1 input, continue.
        if len(node.input) > 1:
            continue
        # If this node is output node, set its previous node as output nodes.
        if helper.find_output_by_name(g, node.output[0]) is not None:
            todel_output = helper.find_output_by_name(g, node.output[0])
            the_input_value = helper.find_value_by_name(g, node.input[0])
            g.output.remove(todel_output)
            g.output.extend([the_input_value])
            node_to_remove.append(node)
            continue
        # Replace the parents in all the following nodes
        following_nodes = helper.find_following_nodes_by_input_value_name(
            g, node.output[0]
        )
        for following_node in following_nodes:
            modhelper.replace_node_input(
                following_node, node.output[0], node.input[0]
            )
        # Delete value info
        value_between = helper.find_value_by_name(g, node.output[0])
        try:
            g.value_info.remove(value_between)
        except Exception:
            print("No value info to delete while eliminating identity layers.")
        # Node is waiting for elimination
        node_to_remove.append(node)
    for node in node_to_remove:
        g.node.remove(node)


def eliminate_nop_Maxpool_and_AveragePool(g):
    """
    Eliminate do nothing MaxPool and AveragePool layers.
    Those layers have valid padding, 1x1 kernel and [1,1] strides.

    :param g: the onnx graph
    """
    node_to_remove = []
    for node in g.node:
        if node.op_type != "MaxPool" and node.op_type != "AveragePool":
            continue
        # If this node is actually working, continue.
        kernel = helper.get_list_attribute_by_name(node, "kernel_shape", "int")
        pads = helper.get_list_attribute_by_name(node, "pads", "int")
        strides = helper.get_list_attribute_by_name(node, "strides", "int")
        if kernel != [1, 1] or pads != [0, 0, 0, 0] or strides != [1, 1]:
            continue
        # If this node is the output, set its previous node as output nodes.
        if helper.find_output_by_name(g, node.output[0]) is not None:
            todel_output = helper.find_output_by_name(g, node.output[0])
            the_input_value = helper.find_value_by_name(g, node.input[0])
            g.output.remove(todel_output)
            g.output.extend([the_input_value])
            node_to_remove.append(node)
            continue
        # Replace the parents in all the following nodes
        following_nodes = helper.find_following_nodes_by_input_value_name(
            g, node.output[0]
        )
        for following_node in following_nodes:
            modhelper.replace_node_input(
                following_node, node.output[0], node.input[0]
            )
        # Delete value info
        value_between = helper.find_value_by_name(g, node.output[0])
        try:
            g.value_info.remove(value_between)
        except Exception:
            print("No value info to delete while eliminating identity layers.")
        # Node is waiting for elimination
        node_to_remove.append(node)
    for node in node_to_remove:
        g.node.remove(node)


def eliminate_trivial_maxpool(g):
    node_to_del = []
    for node in g.node:
        if node.op_type != "MaxPool":
            continue
        pads = None
        strides = None
        dilation = None
        kernel_shape = None
        for att in node.attribute:
            if att.name == "pads":
                pads = list(att.ints)
            elif att.name == "strides":
                strides = list(att.ints)
            elif att.name == "kernel_shape":
                kernel_shape = list(att.ints)
            elif att.name == "dilation":
                dilation = list(att.ints)
            else:
                pass
        if pads and any([pad != 0 for pad in pads]):
            continue
        if strides and any([stride != 1 for stride in strides]):
            continue
        if dilation and any([dila != 1 for dila in dilation]):
            continue
        if any([dim != 1 for dim in kernel_shape]):
            continue

        node_to_del.append(node)

        next_nodes = helper.find_nodes_by_input_name(g, node.output[0])

        if next_nodes[0] is None:
            output_value = helper.find_output_by_name(g, node.output[0])
            if not output_value:
                continue
            else:
                pre_val_info = helper.find_value_by_name(g, node.input[0])
                g.output.extend([pre_val_info])
                g.output.remove(output_value)

        for next_node in next_nodes:
            modhelper.replace_node_input(
                next_node, node.output[0], node.input[0]
            )

        next_val_info = helper.find_value_by_name(g, node.output[0])
        g.value_info.remove(next_val_info)

    while node_to_del:
        g.node.remove(node_to_del.pop())

    other.topological_sort(g)


def eliminate_empty_value_infos(g):
    to_remove = []
    for value_info in g.value_info:
        if len(value_info.type.tensor_type.shape.dim) == 0:
            to_remove.append(value_info)
    for value_info in to_remove:
        g.value_info.remove(value_info)


def eliminate_nop_pads(g):
    node_to_remove = []
    for node in g.node:
        if node.op_type != "Pad":
            continue
        # Check if the Pad is empty or not
        pads_node = helper.find_node_by_output_name(g, node.input[1])
        pads_np = helper.constant_to_numpy(pads_node)
        all_zero = True
        for value in pads_np:
            if value != 0:
                all_zero = False
        if not all_zero:
            continue
        # If this node is the output, set its previous node as output nodes.
        if helper.find_output_by_name(g, node.output[0]) is not None:
            todel_output = helper.find_output_by_name(g, node.output[0])
            g.output.remove(todel_output)
            if helper.find_output_by_name(g, node.input[0]) is None:
                the_input_value = helper.find_value_by_name(g, node.input[0])
                if the_input_value is not None:
                    g.output.extend([the_input_value])
        # Replace the parents in all the following nodes
        following_nodes = helper.find_following_nodes_by_input_value_name(
            g, node.output[0]
        )
        for following_node in following_nodes:
            modhelper.replace_node_input(
                following_node, node.output[0], node.input[0]
            )
        # Delete value info
        value_between = helper.find_value_by_name(g, node.output[0])
        try:
            g.value_info.remove(value_between)
        except Exception:
            helper.logger.info(
                "No value info to delete while eliminating identity layers."
            )
        # Node is waiting for elimination
        node_to_remove.append(node)
    for node in node_to_remove:
        g.node.remove(node)


def eliminate_trivial_elementwise_calculation(g):
    """Eliminate Add, Sub, Mul, Sub nodes which do nothing."""
    node_to_remove = []
    for node in g.node:
        weight_node = None
        if node.op_type == "Add" or node.op_type == "Sub":
            # For add and sub, check if the weights are 0s.
            weight_node = helper.find_node_by_output_name(g, node.input[1])
            if weight_node is None or weight_node.op_type != "Constant":
                continue
            weight_np = helper.constant_to_numpy(weight_node)
            if np.any(weight_np):
                continue
        elif node.op_type == "Mul" or node.op_type == "Div":
            # For Mul and Div, check if the weights are 1s.
            weight_node = helper.find_node_by_output_name(g, node.input[1])
            if weight_node is None or weight_node.op_type != "Constant":
                continue
            weight_np = helper.constant_to_numpy(weight_node)
            weight_np = weight_np - 1
            if np.any(weight_np):
                continue
        else:
            # For other nodes, just skip
            continue
        # Remove the node
        node_to_remove.append(node)
        output_value_info = helper.find_value_by_name(g, node.output[0])
        if output_value_info is not None:
            g.value_info.remove(output_value_info)
        # Replace next node input if any.
        following_nodes = helper.find_following_nodes_by_input_value_name(
            g, node.output[0]
        )
        for following_node in following_nodes:
            modhelper.replace_node_input(
                following_node, node.output[0], node.input[0]
            )
        todel_output = helper.find_output_by_name(g, node.output[0])
        if todel_output is not None:
            g.output.remove(todel_output)
            previous_output = helper.find_output_by_name(g, node.input[0])
            if previous_output is None:
                the_input_value = helper.find_value_by_name(g, node.input[0])
                g.output.extend([the_input_value])
        # Delete the constant node if it is not used by other nodes
        constant_following_nodes = (
            helper.find_following_nodes_by_input_value_name(
                g, weight_node.output[0]
            )
        )
        if len(constant_following_nodes) == 1:
            node_to_remove.append(weight_node)
            output_value_info = helper.find_value_by_name(
                g, weight_node.output[0]
            )
            if output_value_info is not None:
                g.value_info.remove(output_value_info)
    for node in node_to_remove:
        g.node.remove(node)


def eliminate_nop_cast(g):
    """Eliminate do nothing Cast nodes."""
    node_to_remove = []
    for node in g.node:
        if node.op_type != "Cast":
            continue
        # Get input value_info
        input_value = helper.find_value_by_name(g, node.input[0])
        if input_value is None:
            helper.logger.debug(
                f"Cannot find the input value_info for Cast node {node.name}. "
                "Skip elimination check."
            )
            continue
        # Get output value_info
        output_value = helper.find_value_by_name(g, node.output[0])
        if output_value is None:
            output_value = helper.find_output_by_name(g, node.output[0])
        if output_value is None:
            helper.logger.debug(
                f"Cannot find the output value_info for Cast node {node.name}."
                " Skip elimination check."
            )
            continue
        # Compare the type.
        if (
            input_value.type.tensor_type.elem_type
            != output_value.type.tensor_type.elem_type
        ):
            continue
        # If this node is the output, set its previous node as output nodes.
        if helper.find_output_by_name(g, node.output[0]) is not None:
            todel_output = helper.find_output_by_name(g, node.output[0])
            g.output.remove(todel_output)
            if helper.find_output_by_name(g, node.input[0]) is None:
                the_input_value = helper.find_value_by_name(g, node.input[0])
                if the_input_value is not None:
                    g.output.extend([the_input_value])
        # Replace the parents in all the following nodes
        following_nodes = helper.find_following_nodes_by_input_value_name(
            g, node.output[0]
        )
        for following_node in following_nodes:
            modhelper.replace_node_input(
                following_node, node.output[0], node.input[0]
            )
        # Delete value info
        value_between = helper.find_value_by_name(g, node.output[0])
        if value_between is not None:
            g.value_info.remove(value_between)
        # Node is waiting for elimination
        node_to_remove.append(node)
    for node in node_to_remove:
        g.node.remove(node)
