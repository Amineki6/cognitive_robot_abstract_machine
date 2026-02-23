from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
from typing_extensions import Union, Optional, Type, Any, Iterable

from krrood.entity_query_language.entity import and_
from krrood.entity_query_language.symbolic import SymbolicExpression
from semantic_digital_twin.reasoning.robot_predicates import is_pose_free_for_robot
from semantic_digital_twin.robots.abstract_robot import Camera
from ..base import ActionDescription, DescriptionType
from ...motions.robot_body import LookingMotion
from ...motions.navigation import MoveMotion
from ....config.action_conf import ActionConfig
from ....datastructures.partial_designator import PartialDesignator
from ....datastructures.pose import PoseStamped
from ....failures import LookAtGoalNotReached
from ....failures import NavigationGoalNotReachedError
from ....language import SequentialPlan
from ....validation.error_checkers import PoseErrorChecker


@dataclass
class NavigateAction(ActionDescription):
    """
    Navigates the Robot to a position.
    """

    target_location: PoseStamped
    """
    Location to which the robot should be navigated
    """

    keep_joint_states: bool = ActionConfig.navigate_keep_joint_states
    """
    Keep the joint states of the robot the same during the navigation.
    """

    def execute(self) -> None:
        return SequentialPlan(
            self.context, MoveMotion(self.target_location, self.keep_joint_states)
        ).perform()

    def pre_condition(self, bound=True) -> SymbolicExpression:
        variables = self.bound_variables if bound else self.unbound_variables
        return and_(
            is_pose_free_for_robot(
                self.robot_view, variables[self.target_location].to_spatial_type()
            ),
            self.robot_view.drive is not None,
        )

    def post_condition(self, bound=True) -> SymbolicExpression:
        variables = self.bound_variables if bound else self.unbound_variables
        return and_(
            np.allclose(
                self.robot_view.root.global_pose,
                self.target_location.to_spatial_type(),
                atol=0.03,
            )
        )

    @classmethod
    def description(
        cls,
        target_location: DescriptionType[PoseStamped],
        keep_joint_states: DescriptionType[
            bool
        ] = ActionConfig.navigate_keep_joint_states,
    ) -> PartialDesignator[NavigateAction]:
        return PartialDesignator[NavigateAction](
            NavigateAction,
            target_location=target_location,
            keep_joint_states=keep_joint_states,
        )


@dataclass
class LookAtAction(ActionDescription):
    """
    Lets the robot look at a position.
    """

    target: PoseStamped
    """
    Position at which the robot should look, given as 6D pose
    """

    camera: Camera = None
    """
    Camera that should be looking at the target
    """

    def execute(self) -> None:
        camera = self.camera or self.robot_view.get_default_camera()
        SequentialPlan(
            self.context, LookingMotion(target=self.target, camera=camera)
        ).perform()

    def validate(
        self, result: Optional[Any] = None, max_wait_time: Optional[timedelta] = None
    ):
        """
        Check if the robot is looking at the target location by spawning a virtual object at the target location and
        creating a ray from the camera and checking if it intersects with the object.
        """
        return

    @classmethod
    def description(
        cls,
        target: DescriptionType[PoseStamped],
        camera: DescriptionType[Camera] = None,
    ) -> PartialDesignator[LookAtAction]:
        return PartialDesignator[LookAtAction](
            LookAtAction, target=target, camera=camera
        )


NavigateActionDescription = NavigateAction.description
LookAtActionDescription = LookAtAction.description
