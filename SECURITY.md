# Security Policy

## Supported Versions

The following versions of Adaptive Rate Limiter are currently supported with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you discover a security vulnerability in Adaptive Rate Limiter, please report it privately by emailing:

**report@sbang.dev**

### Response Timeline

- **Acknowledgment**: Within 48 hours of receiving your report
- **Initial Assessment**: Within 7 days, we will provide a status update on the vulnerability
- **Resolution Timeline**: Varies based on severity; critical issues are prioritized for immediate patching

### What to Include in Your Report

To help us investigate and resolve the issue quickly, please include:

1. **Description**: A clear description of the vulnerability
2. **Affected Component**: Which part of the library is affected (e.g., scheduler, backends, streaming)
3. **Reproduction Steps**: Detailed steps to reproduce the vulnerability
4. **Proof of Concept**: Code snippets or scripts demonstrating the issue (if available)
5. **Impact Assessment**: Your assessment of the potential impact and severity
6. **Suggested Fix**: Any recommendations for remediation (optional)

## Safe Harbor

We consider security research conducted in accordance with this policy to be:

- **Authorized** in accordance with the Computer Fraud and Abuse Act (CFAA) and similar laws
- **Exempt** from DMCA restrictions on circumventing technological measures
- **Lawful**, helpful, and conducted in good faith

We will not pursue legal action against researchers who:

- Act in good faith to avoid privacy violations, data destruction, and service interruption
- Report vulnerabilities directly to us and allow reasonable time for resolution
- Do not exploit vulnerabilities beyond what is necessary to demonstrate the issue
- Do not publicly disclose the vulnerability before we have had an opportunity to address it

## Scope

### In Scope

- The Adaptive Rate Limiter library source code (`src/adaptive_rate_limiter/`)
- Official documentation that could lead to security issues
- Configuration handling that could expose sensitive information

### Out of Scope

- Third-party dependencies (please report these to the respective maintainers)
- Infrastructure hosting the repository (GitHub security issues)
- Social engineering attacks
- Denial of service attacks that don't reveal an underlying vulnerability
- Issues in applications built using this library (unless the root cause is in the library itself)

## Security Best Practices for Users

When using Adaptive Rate Limiter in production:

1. **Redis Backend**: Use authentication and TLS when connecting to Redis
2. **Configuration**: Avoid logging or exposing rate limit configuration that could aid attackers
3. **Updates**: Keep the library updated to receive security patches
4. **Monitoring**: Monitor for unusual rate limit patterns that could indicate abuse

## Acknowledgments

We appreciate the security research community's efforts in helping keep Adaptive Rate Limiter secure. Researchers who report valid vulnerabilities will be acknowledged in our security advisories (unless they prefer to remain anonymous).
