"""Visible failure types. Provider/data problems must not become silent defaults."""

class HockeyMoneylineError(Exception):
    """Base project error."""

class AcquisitionError(HockeyMoneylineError):
    """Historical-source fetch failed or returned an unusable payload."""

class NHLApiError(HockeyMoneylineError):
    """Bounded NHL Web API read or payload validation failed."""

class IdentityError(HockeyMoneylineError):
    """MoneyPuck ↔ NHL identity mapping was ambiguous or inconsistent."""

class SchemaError(HockeyMoneylineError):
    """Normalized historical-game contract was violated."""

class FeatureError(HockeyMoneylineError):
    """Feature construction failed."""

class EvaluationError(HockeyMoneylineError):
    """Chronological evaluation could not be completed from the available games."""


class PredictionError(HockeyMoneylineError):
    """Pregame scoring failed or required source/history state was unavailable."""


class CurrentSeasonPriorSourceError(HockeyMoneylineError):
    """Current-season prior-state source is missing, stale, incomplete, or conflicting."""


class HostedRefreshError(HockeyMoneylineError):
    """Hosted refresh input was rejected, or a caller used a bundle that is not ready."""

    def __init__(self, message: str, *, code: str = "hosted_refresh_rejected") -> None:
        super().__init__(message)
        self.code = code


class ProjectHqError(HockeyMoneylineError):
    """Project HQ lifecycle command or sender envelope was invalid."""


class MarketMathError(HockeyMoneylineError):
    """Two-sided decimal moneyline inputs cannot support the market-math contract."""


class RecommendationContractError(HockeyMoneylineError):
    """Odds-edge recommendation payload violated the versioned output contract."""


class ForwardOddsArchiveError(HockeyMoneylineError):
    """Forward-odds snapshot was invalid, conflicted, or an assist failed closed."""
