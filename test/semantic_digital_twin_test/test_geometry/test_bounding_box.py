from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import BoundingBox


def test_bounding_box_transform_same_frame(pr2_apartment_state_reset):
    bb = BoundingBox(
        -1,
        -1,
        -1,
        1,
        1,
        1,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            reference_frame=pr2_apartment_state_reset.root
        ),
    )

    new_origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1, 1, 1, reference_frame=pr2_apartment_state_reset.root
    )

    assert bb.min_x == -1
    assert bb.max_x == 1
    assert bb.min_y == -1
    assert bb.max_y == 1
    assert bb.min_z == -1
    assert bb.max_z == 1
    assert bb.origin.to_position().to_np().tolist() == [0, 0, 0, 1]

    new_origin_bb = bb.transform_to_origin(new_origin)

    assert new_origin_bb.min_x == 0
    assert new_origin_bb.max_x == 2
    assert new_origin_bb.min_y == 0
    assert new_origin_bb.max_y == 2
    assert new_origin_bb.min_z == 0
    assert new_origin_bb.max_z == 2
    assert new_origin_bb.origin.to_position().to_np().tolist() == [1, 1, 1, 1]


def test_bounding_box_transform_different_frame(pr2_apartment_state_reset):
    bb = BoundingBox(0, 0, 0, 1, 1, 1, pr2_apartment_state_reset.root.global_pose)

    new_origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        0,
        0,
        0,
        reference_frame=pr2_apartment_state_reset.get_body_by_name("base_footprint"),
    )

    assert bb.min_x == 0
    assert bb.max_x == 1
    assert bb.min_y == 0
    assert bb.max_y == 1
    assert bb.min_z == 0
    assert bb.max_z == 1
    assert bb.origin.to_position().to_np().tolist() == [0, 0, 0, 1]

    new_origin_bb = bb.transform_to_origin(new_origin)

    assert new_origin_bb.min_x == 0
    assert new_origin_bb.max_x == 1
    assert new_origin_bb.min_y == 0
    assert new_origin_bb.max_y == 1
    assert new_origin_bb.min_z == 0
    assert new_origin_bb.max_z == 1
    assert new_origin_bb.origin.to_position().to_np().tolist() == [0, 0, 0, 1]
